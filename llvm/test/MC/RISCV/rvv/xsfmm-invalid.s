# RUN: not llvm-mc -triple=riscv32 --mattr=+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a64f %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK
# RUN: not llvm-mc -triple=riscv64 --mattr=+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a64f %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK

sf.vsettnt a0, a1, a2, e128, w1 # CHECK: :[[@LINE]]:20: error: operand must be e[8|16|16alt|32|64],w[1|2|4]

sf.vsettnt a0, a1, a2, e8alt, w1 # CHECK: :[[@LINE]]:20: error: operand must be e[8|16|16alt|32|64],w[1|2|4]

sf.mm.f.f mt1, v8, v9 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction

sf.mm.e5m2.e5m2 mt2, v8, v9 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction

sf.mm.e5m2.e4m3 mt6, v8, v9 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction

sf.mm.e4m3.e5m2 mt10, v8, v9 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction

sf.mm.e4m3.e4m3 mt14, v8, v9 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction

sf.mm.u.u mt1, v8, v9 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction

sf.mm.s.u mt2, v8, v9 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction

sf.mm.u.s mt3, v8, v9 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction

sf.mm.s.s mt1, v8, v9 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction

sf.p2mm.u.u mt1, v8, v9 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction

sf.p2mm.s.u mt1, v8, v9 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction

sf.p2mm.u.s mt1, v8, v9 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction

sf.p2mm.s.s mt1, v8, v9 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
