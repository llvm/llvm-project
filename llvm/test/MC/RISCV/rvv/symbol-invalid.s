# RUN: not llvm-mc --triple=riscv64 -mattr=+v < %s 2>&1 | FileCheck %s

reldef:

.global undef

## simm5_plus1
vmsgeu.vi v3, v4, reldef, v0.t   # CHECK: :[[@LINE]]:19: error: immediate must be in the range [-15, 16]
vmsgeu.vi v3, v4, reldef-., v0.t # CHECK: :[[@LINE]]:19: error: immediate must be in the range [-15, 16]
vmsgeu.vi v3, v4, undef, v0.t    # CHECK: :[[@LINE]]:19: error: immediate must be in the range [-15, 16]
vmsgeu.vi v3, v4, latedef, v0.t  # CHECK: :[[@LINE]]:19: error: immediate must be in the range [-15, 16]


## simm5
vadd.vi v4, v5, reldef, v0.t   # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [-16, 15]
vadd.vi v4, v5, reldef-., v0.t # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [-16, 15]
vadd.vi v4, v5, undef, v0.t    # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [-16, 15]
vadd.vi v4, v5, latedef, v0.t  # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [-16, 15]

.set latedef, 1
