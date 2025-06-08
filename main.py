import random
import math
import numpy as np

# Визначення функції Сфери
def sphere_function(x):
    return sum(xi ** 2 for xi in x)


# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    current_point = tuple([random.uniform(b[0], b[1]) for b in bounds])
    current_value = func(current_point)

    def get_neighbors(current, bounds, step_size=0.1):
        x, y = current
        result = []

        new_x = x + step_size
        if new_x <= bounds[0][1]:
            result.append((new_x, y))
        new_x = x - step_size
        if new_x >= bounds[0][0]:
            result.append((new_x, y))

        new_y = y + step_size
        if new_y <= bounds[1][1]:
            result.append((x, new_y))
        new_y = y - step_size
        if new_y >= bounds[1][0]:
            result.append((x, new_y))

        return result

    for _ in range(iterations):
        neighbors = get_neighbors(current_point, bounds)

        # Пошук найкращого сусіда
        next_point = None
        next_value = np.inf

        for neighbor in neighbors:
            value = func(neighbor)
            if value < next_value:
                next_point = neighbor
                next_value = value

        # Якщо не вдається знайти кращого сусіда обо досягли необхідного наближення — зупиняємось
        if next_value >= current_value or abs(current_value - next_value) < epsilon:
            break

        # Переходимо до кращого сусіда
        current_point, current_value = next_point, next_value

    return current_point, current_value


# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    current_point = [random.uniform(b[0], b[1]) for b in bounds]
    current_value = func(current_point)

    def get_random_neighbor(current, step_size=0.5):
        x, y = current
        new_x = x + random.uniform(-step_size, step_size)
        new_y = y + random.uniform(-step_size, step_size)
        return (new_x, new_y)
        
    for _ in range(iterations):
        # Отримання випадкового сусіда
        new_point = get_random_neighbor(current_point)
        if (
            new_point[0] > bounds[0][1]
            or new_point[0] < bounds[0][0]
            or new_point[1] > bounds[1][1]
            or new_point[1] < bounds[1][0]
        ):
            break # відкидданя сусіда за межами інтервалу

        new_value = func(new_point)

        # Перевірка умови переходу
        if new_value < current_value or abs(current_value - new_value) < epsilon:
            current_point, current_value = new_point, new_value

    return current_point, current_value


# Simulated Annealing
def simulated_annealing(func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6):
    current = [random.uniform(b[0], b[1]) for b in bounds]
    current_value = func(current)
    best, best_value = current, current_value

    for _ in range(iterations):
        temp *= cooling_rate
        if temp < epsilon:
            break

        neighbor = [
            max(bounds[i][0], min(bounds[i][1], current[i] + random.uniform(-0.1, 0.1)))
            for i in range(len(bounds))
        ]
        neighbor_value = func(neighbor)

        if neighbor_value < current_value or random.random() < math.exp(
            (current_value - neighbor_value) / temp
        ):
            current, current_value = neighbor, neighbor_value

        if current_value < best_value:
            best, best_value = current, current_value

    return best, best_value


if __name__ == "__main__":
    # Межі для функції
    bounds = [(-5, 5), (-5, 5)]

    # Виконання алгоритмів
    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds)
    print("Розв'язок:", hc_solution, "Значення:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(sphere_function, bounds)
    print("Розв'язок:", rls_solution, "Значення:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
    print("Розв'язок:", sa_solution, "Значення:", sa_value)
