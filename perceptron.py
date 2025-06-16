import matplotlib.pyplot as plt
import os

path = 'C:/Users/katar/PycharmProjects/PythonProject2/perceptron'

class Perceptron:
    def __init__(self, x1, x2, y_d, bias, alpha, w1, w2, end):
        self.x1 = x1
        self.x2 = x2
        self.y_d = y_d
        self.bias = bias
        self.alpha = alpha
        self.end = end
        self.w1 = w1
        self.w2 = w2

    def train(self):
        epoch_num = 1
        def epoch():
            nonlocal epoch_num
            err = []
            line1 = f'Epoka {epoch_num}\n'
            line2 = f'{"x1":^3} | {"x2":^3} | {"w1":^5} | {"w2":^5} | {"Yd":^3} | {"Yp":^3} | {"e":^3} | {"dw1":^6} | {"dw2":^6}\n'
            print(line1)
            print(line2)
            with open('training.txt', 'a') as f:
                f.write(line1)
                f.write(line2)
            for i in range(len(self.y_d)):
                w_sum = self.x1[i]*self.w1 + self.x2[i]*self.w2 - self.bias
                if w_sum >= self.bias:
                    y_p = 1
                else:
                    y_p = -1
                e = self.y_d[i] - y_p
                err.append(e)
                d_w1 = self.alpha * self.x1[i] * e
                d_w2 = self.alpha * self.x2[i] * e
                lines = f'{self.x1[i]:^3} | {self.x2[i]:^3} | {self.w1:^5.2f} | {self.w2:^5.2f} | {self.y_d[i]:^3} | {y_p:^3} | {e:^3} | {d_w1:^6.2f} | {d_w2:^6.2f}\n'
                print(lines)
                with open('training.txt', 'a') as f:
                    f.write(lines)
                self.w1 += d_w1
                self.w2 += d_w2
            print('\n')
            with open('training.txt', 'a') as f:
                f.write('\n')
            return err

        errors_tab = []
        mean_err = float('inf')
        while mean_err != 0 and epoch_num <= self.end:
            error = epoch()
            err_sum = 0
            for item in error:
                err_sum += abs(item)
            mean_err = err_sum/len(error)
            errors_tab.append(mean_err)
            epoch_num += 1
        return errors_tab, epoch_num

    def plots(self, errors, epochs):
        #Błąd na epokę
        plt.figure()
        plt.plot([i+1 for i in range(epochs-1)], errors)
        plt.xticks([i+1 for i in range(epochs-1)])
        plt.xlabel('Epoka')
        plt.ylabel('Błąd')
        plt.title('Średni błąd na epokę')
        plt.savefig('error.jpg')

        #Separowalność - hardcoded bo wyliczanie ze wzoru który znalazłam nie działało :(
        #Możliwe że perceptron sobie to inaczej oddziela ale przynajmniej widać że problem separowalny?
        sep_x = (2.5 + 8) / 2
        sep_y = (2.1 + 6.4) / 2
        plt.figure()
        plt.scatter(self.x1, self.x2)
        plt.axline((sep_x, sep_y), slope = -1)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Separowalność')
        plt.savefig('sep.jpg')

os.chdir(path)

#Initializing and running
p = Perceptron(x1 = (1, 9.4, 2.5, 8), x2 = (1, 6.4, 2.1, 7.7), y_d = (1, -1, 1, -1), bias = -1.5, alpha = 0.55, w1 = 1, w2 = 1, end = 10)
error_table, number_of_epochs = p.train()
p.plots(error_table, number_of_epochs)