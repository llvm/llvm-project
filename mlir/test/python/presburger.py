from mlir import presburger
import numpy as np

"""
Test the following integer relation

x + 2y = 8
x - y <= 1
y >= 3
"""
# eqs = np.asarray([[1, 2, -8]], dtype=np.int64)
# ineqs = np.asarray([[1, -1, -1]], dtype=np.int64)
# relation = presburger.IntegerRelation(ineqs, eqs, 2, 0)
# print(relation)
# relation.add_inequality([0, 1, -3])
# print(relation)
# t = relation.equalities()
# print(t)
# print(relation.num_constraints)
# print(relation.num_inequalities)
# print(relation.num_equalities)
# print(relation.num_domain_vars)
# print(relation.num_range_vars)
# print(relation.num_symbol_vars)
# print(relation.num_local_vars)
# print(relation.num_columns)

# eq_first_row = relation.get_equality(0)
# print(eq_first_row)
# ineq_second_row = relation.get_inequality(1)
# print(ineq_second_row)

# eq_coefficients = relation.equalities()
# print(eq_coefficients[0, 1])
# ineq_coefficients = relation.inequalities()
# print(ineq_coefficients[1, 1])

"""
Test intersection

Relation A

x + y <= 6

Relation B

x>=2
"""
# print("-------")
# eqs_a = np.asarray([[0, 0, 0]], dtype=np.int64)
# ineqs_a = np.asarray([[-1, -1, 6]], dtype=np.int64)
# relation_a = presburger.IntegerRelation(ineqs_a, eqs_a, 2, 0)
# print(relation_a)

# eqs_b = np.asarray([[0, 0, 0]], dtype=np.int64)
# ineqs_b = np.asarray([[1, 0, -2]], dtype=np.int64)
# relation_b = presburger.IntegerRelation(ineqs_b, eqs_b, 2, 0)
# print(relation_b)

# a_b_intersection = relation_a.intersect(relation_b)
# print(a_b_intersection)

# print(a_b_intersection.num_vars)
# print(a_b_intersection.get_var_kind_at(1))

"""
y = 2x
x <= 5
0 <= x

"""
# eqs = np.asarray([[-2, 1, 0]], dtype=np.int64)
# ineqs = np.asarray([[-1, 0, 5], [1, 0, 0]], dtype=np.int64)
# relation = presburger.IntegerRelation(ineqs, eqs, 1, 1)
# print(relation)
# print(relation.num_vars)
# t = relation.get_var_kind_at(1)
# print(t, type(t))
# print(relation.append_var(presburger.VariableKind.Range))
# print(relation)


"""
x+y = 20
x <= 20
x >= 6
y <= 14
y >= 5
"""

eqs = np.asarray([[1, 1, -20]], dtype=np.int64)
ineqs = np.asarray([[-1, 0, 20], [1, 0, -6], [0, 1, -5], [0, -1, 14]], dtype=np.int64)
relation = presburger.IntegerRelation(ineqs, eqs, 2, 0)
print(relation)
lex_min = relation.find_integer_lex_min()

print(lex_min)
print(lex_min.is_unbounded())
print(lex_min.is_bounded())
print(lex_min.get_integer_point())

integer_sample = relation.find_integer_sample()
print(integer_sample)

int_volume = relation.compute_volume()
print(int_volume)

print(relation.contains_point([23, 7]))