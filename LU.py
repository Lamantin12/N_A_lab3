from numpy import zeros, matrix


class lu_decomposion:
    def __init__(self, matrix):
        self.origin_matrix = matrix
        self.lu = ()
        self.partial = None
        self.change_origin = None
        self.solution = None

    def fit(self, **kwargs):
        self.change_origin = kwargs['change_origin']
        self.partial = kwargs['partial_pivoting']
        if self.partial:
            if self.change_origin:
                pass
            else:
                pass  #
        else:
            if self.change_origin:
                self.lu_without_partial_change_origin()
            else:
                pass

    def lu_without_partial_change_origin(self):
        n = self.origin_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                t = self.origin_matrix[j, i] / self.origin_matrix[i, i]
                self.origin_matrix[j, i:] = self.origin_matrix[j, i:] - self.origin_matrix[i, i:] * t
                self.origin_matrix[j, i] = t
        return 'method fitted, origin matrix changed'

    def solve(self, vector):
        if self.partial:
            if self.change_origin:
                pass
            else:
                pass  #
        else:
            if self.change_origin:
                n = self.origin_matrix.shape[0]
                l = self.get_l_from_lu()
                u = self.get_u_from_lu()

                y = matrix(zeros((n, 1)))
                self.solution = matrix(zeros((n, 1)))

                for i in range(n):
                    y[i] = vector[i] - l[i, :i] * y[:i]

                for i in range(n - 1, -1, -1):
                    self.solution[i] = (y[i] - u[i, i:] * self.solution[i:]) / u[i][i]
                print('Решение:', self.solution)
            else:
                pass

    def get_l_from_lu(self):
        n = self.origin_matrix.shape[0]
        l = zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if j != i:
                    l[j, i] = self.origin_matrix[j, i]
                else:
                    l[j, i] = 1

        return l

    def get_u_from_lu(self):
        n = self.origin_matrix.shape[0]
        u = zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                u[i, j] = self.origin_matrix[i, j]
        return u

    def display_results(self):
        if self.solution is not None:
            if self.partial:
                if self.change_origin:
                    pass
                else:
                    pass  #
            else:
                if self.change_origin:
                    print('Полученная измененная оригинальная матрица', self.origin_matrix)
                else:
                    pass
        else:
            print('YOLO')
