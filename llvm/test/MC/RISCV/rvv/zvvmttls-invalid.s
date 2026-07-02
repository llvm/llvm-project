# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zvvmttls %s 2>&1 \
# RUN:        | FileCheck %s

vmttl.v v8, (a0), a1, L0
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmttl.v v8, (a0), a1, L3
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmttl.v v8, (a0), a1, L128
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmttl.v v8, (a0), a1, L4, v1.t
# CHECK: error: operand must be v0.t

vmttl.v v8, (a0), a1, v0.t, L4
# CHECK: error: invalid operand for instruction

vmtts.v v12, (a0), a1, L0
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmtts.v v12, (a0), a1, L3
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmtts.v v12, (a0), a1, L128
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmtts.v v12, (a0), a1, L4, v1.t
# CHECK: error: operand must be v0.t

vmtts.v v12, (a0), a1, v0.t, L4
# CHECK: error: invalid operand for instruction
