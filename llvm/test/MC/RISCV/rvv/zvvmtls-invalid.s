# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zvvmtls %s 2>&1 \
# RUN:        | FileCheck %s

vmtl.v v8, (a0), a1, L0
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmtl.v v8, (a0), a1, L3
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmtl.v v8, (a0), a1, L128
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmtl.v v8, (a0), a1, L4, v1.t
# CHECK: error: operand must be v0.t

vmtl.v v8, (a0), a1, v0.t, L4
# CHECK: error: invalid operand for instruction

vmts.v v12, (a0), a1, L0
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmts.v v12, (a0), a1, L3
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmts.v v12, (a0), a1, L128
# CHECK: error: operand must be L1, L2, L4, L8, L16, L32, or L64

vmts.v v12, (a0), a1, L4, v1.t
# CHECK: error: operand must be v0.t

vmts.v v12, (a0), a1, v0.t, L4
# CHECK: error: invalid operand for instruction
