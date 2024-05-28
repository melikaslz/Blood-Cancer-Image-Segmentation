import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math

def initial_population(n_population,n_colors):
    pop=[]
    for i in range(n_population):
        answer=[]
        for j in range(n_colors):
            r=random.randrange(i,255)
            g=random.randrange(i,255)
            b=random.randrange(i,255)
            color=[r,g,b]
            answer.append(color)
        answer.sort()
        pop.append(answer)
    return pop

def fitness(image,n_colors,answer):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
    result = 0
    for i in range(resized_img.shape[0]):
        for j in range(resized_img.shape[1]):
            pixel = resized_img[i][j]
            dist = math.inf
            for color in answer:
                current_dist = 0
                for k in range(3):
                    current_dist += (pixel[k]-color[k])**2
                if current_dist<dist:
                    dist = current_dist
            result+= math.sqrt(dist)
    result /= (resized_img.shape[0]*resized_img.shape[1])
    return float("{:.4f}".format(result))


def selection(image ,n_color,pop):
    size = len(pop)
    pop = sorted(pop,key= lambda x:fitness(image,n_color, x))
    return pop[:size//2]


def crossover(pop, n_color):
    new_pop=[]
    for i in range(len(pop)):
        par1,par2 = random.sample(pop,2)
        if pop.index(par1)>pop.index(par2):
            father = par1
            mother = par2
        else:
            mother = par1
            father = par2
        child = [None for i in range(n_color)]
        for j in range(n_color):
            r= mother[j][0]*2/3+father[j][0]*1/3
            g= mother[j][1]*2/3+father[j][1]*1/3
            b= mother[j][2]*2/3+father[j][2]*1/3
            child[j]=[int(r),int(g),int(b)]
        new_pop.append(mother)
        new_pop.append(child)
    return new_pop

def mutation(pop , p, n_color):
    new_pop =[]
    for answer in pop:
        new_ans = list(answer).copy()
        if random.random()<p:
            color = random.randrange(n_color)
            r = 255- answer[color][0]
            g = 255- answer[color][1]
            b = 255 - answer[color][2]
            new_ans[color]=[r,g,b]
        new_pop.append(new_ans)
    return new_pop

def GA(image,n_generation,n_population,n_color,p_mutate):
    best_fit = math.inf
    pop = initial_population(n_population,n_color)
    result = pop[0]
    for i in range(n_generation):
        pop = selection(image,n_color,pop)
        pop = crossover(pop,n_color)
        pop = mutation(pop,p_mutate,n_color)
        current_fit = fitness(image , n_color, pop[0])
        if current_fit < best_fit:
            best_fit = current_fit
            print('in genertion {} best fitness is: {}'.format(i+1, best_fit))
            print('The answer is :{}'.format(pop[0]))
            result = pop[0]
    return result

def psnr(in_img,out_img):
    I = cv2.cvtColor(in_img, cv2.COLOR_RGB2GRAY)
    O = cv2.cvtColor(out_img, cv2.COLOR_RGB2GRAY)
    mse = np.mean((I - O) ** 2)
    if mse == 0:
        return 100
    #255 tedade pixel haast
    return 10 * np.log10(255**2 / mse)

def Recolored(image, colors):
    out_image = np.copy(image)
    for i in range(len(image)):
        for j in range(len(image[0])):
            pixel = image[i][j]
            dist = math.inf
            for color in colors:
                current_dist = 0
                for k in range(3):
                    current_dist += (pixel[k]-color[k])**2
                if current_dist < dist:
                    dist = current_dist
                    temp = color
            out_image[i][j] = temp
    #plt.imshow(out_image)
    #plt.show()
    return out_image

def ShowImage(image,title):
    cv2.imshow(title,image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

