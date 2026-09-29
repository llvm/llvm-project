# A Polly generated test case that would cause a failure
# in isl_basic_map_reduce_coefficients in an earlier version of isl.
domain: "[p_0, p_1] -> { Stmt2[i0] : i0 >= 0 and ((p_1 < p_0 and 4*floor((2 + p_1)/4) <= p_1 - i0) or (p_1 >= p_0 and 4*floor((p_0 - p_1)/4) < p_0 - p_1 and 4*floor((2 + p_0)/4) <= p_0 - i0) or (p_1 < p_0 and p_1 < 4*floor((2 + p_1)/4) <= 4 + p_1 - i0 and 4*floor((p_0 - p_1)/4) < p_0 - p_1) or (p_1 >= p_0 and 4*floor((p_0 - p_1)/4) < p_0 - p_1 and p_0 < 4*floor((2 + p_0)/4) <= 4 + p_0 - i0) or ((2 + p_0) mod 4 = 6 + p_1 + 4*floor((-1 - p_1)/4) and p_1 >= p_0 and -4 - p_1 + i0 <= 4*floor((-1 - p_1)/4) <= -3 - p_1) or ((2 + p_0) mod 4 = 2 + p_1 + 4*floor((1 - p_1)/4) and p_1 >= p_0 and 4*floor((1 - p_1)/4) >= -4 - p_1 + i0 and 4*floor((1 - p_1)/4) < -p_1) or ((-p_0 + p_1) mod 4 = 0 and p_1 <= -4 + p_0 and p_1 < 4*floor((2 + p_1)/4) <= 4 + p_1 - i0)) }"
child:
  context: "[p_0, p_1] -> { [] : p_0 >= -2147483648 and p_0 <= 2147483647 and p_1 >= -2147483648 and p_1 <= 2147483647 }"
  child:
    schedule: "[p_0, p_1] -> [{ Stmt2[i0] -> [(i0)] }]"
