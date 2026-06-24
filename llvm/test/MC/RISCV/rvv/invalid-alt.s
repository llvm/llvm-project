# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zvfbfa --mattr=+f %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zvfofp8min --mattr=+f %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vsetvli a2, a0, e32alt, m1, ta, ma
# CHECK-ERROR: invalid instruction

vsetvli a2, a0, e64alt, m1, ta, ma
# CHECK-ERROR: invalid instruction
