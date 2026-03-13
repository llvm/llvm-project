# RUN: llvm-mc %s -triple=riscv32 -mattr=+zmmul -M no-aliases 2>&1 \
# RUN:  | FileCheck -check-prefixes=CHECK-INST %s

# CHECK-INST: mul a4, ra, s0
mul a4, ra, s0

# CHECK-INST: mulh ra, zero, zero
mulh x1, x0, x0

# CHECK-INST: mulhsu t0, t2, t1
mulhsu t0, t2, t1

# CHECK-INST: mulhu a5, a4, a3
mulhu a5, a4, a3
