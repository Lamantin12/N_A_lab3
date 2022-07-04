from numpy import zeros, matrix, copy, dot


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
                self.lu_with_partial_no_change_origin()
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
        self.lu = copy(self.origin_matrix)
        return 'method fitted, origin matrix changed'

    def lu_with_partial_no_change_origin(self):
        n = self.origin_matrix.shape[0]
        p = zeros((n, n))
        for i in range(n):
            p[i, i] = 1
        u = copy(self.origin_matrix)

        l = zeros((n, n))
        for i in range(n):
            l[i, i] = 1
        p_temp_in_list = []
        for i in range(n):
            p_temp = zeros((n, n))

            m = -100000
            max_k = i
            for k in range(i, n):
                if u[k, i] > m:
                    m = u[k, i]
                    max_k = k

            swap_col(u, i, max_k)
            p_temp[i, max_k] = 1
            p_temp[max_k, i] = 1
            for p in range(n):
                if (p != i) & (p != max_k):
                    p_temp[p, p] = 1
            p_temp_in_list.append(p_temp)

            for j in range(i + 1, n):
                t = u[j, i] / u[i, i]
                l[j, i] = t
                u[j, i:] = u[j, i:] - u[i, i:] * t
        for i in range(n - 1, -1, -1):
            p = dot(p, p_temp_in_list[i])
        p = p / 4
        self.lu = (l, u, p)
        print('method fitted, created a tuple (L, U, P)')

    def solve(self, vector):
        if self.partial:
            if self.change_origin:
                pass
            else:
                n = vector.shape[0]
                l = self.lu[0]
                u = self.lu[1]
                
                y = matrix(zeros((vector.shape[0], 1)))
                self.solution = matrix(zeros((vector.shape[0], 1)))

                for i in range(n):
                    y[i] = vector[i] - l[i, :i] * y[:i]

                for i in range(n - 1, -1, -1):
                    self.solution[i] = (y[i] - u[i, i + 1:] * self.solution[i + 1:]) / u[i][i]
                self.solution = dot(self.solution.T, self.lu[2])
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
            else:
                pass
        return self.solution

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


def swap_col(mtrx, start_index, last_index):
    mtrx[:, [start_index, last_index]] = mtrx[:, [last_index, start_index]]
