import RustMath as rm

# Tests:
print(rm.dot_product([1, 2, 3], [4, 5, 6]))
print(rm.cross_product([1, 2, 3], [4, 5, 6]))
print(rm.cosine_similarity([1, 2, 3], [4, 5, 6]))
print(rm.vector_magnitude([1, 2, 3]))
print(rm.unit_vector([1, 2, 3]))
print(rm.angle_between([1, 2, 3], [4, 5, 6]))
print(rm.matrix_product([[1, 2], [3, 4]], [[5, 6], [7, 8]]))
print(rm.mean([1, 2, 3, 4, 5]))
print(rm.variance([1, 2, 3, 4, 5]))
print(rm.standard_deviation([1, 2, 3, 4, 5]))
print(rm.softmax([1, 2, 3]))