# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zvfbfa --mattr=+f %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vsetvli a2, a0, e32alt, m1, ta, ma
# CHECK-ERROR: operand must be e[8|8alt|16|16alt|32|64],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e64alt, m1, ta, ma
# CHECK-ERROR: operand must be e[8|8alt|16|16alt|32|64],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]
