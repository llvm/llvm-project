# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfinx -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zfinx %s \
# RUN:     | llvm-objdump --mattr=+zfinx -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -mattr=+zfinx %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s

# CHECK-ASM-AND-OBJ: fcvt.l.s a0, t0, dyn
# CHECK-ASM: encoding: [0x53,0xf5,0x22,0xc0]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.l.s a0, t0, dyn
# CHECK-ASM-AND-OBJ: fcvt.lu.s a1, t1, dyn
# CHECK-ASM: encoding: [0xd3,0x75,0x33,0xc0]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.lu.s a1, t1, dyn
# CHECK-ASM-AND-OBJ: fcvt.s.l t2, a2, dyn
# CHECK-ASM: encoding: [0xd3,0x73,0x26,0xd0]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.s.l t2, a2, dyn
# CHECK-ASM-AND-OBJ: fcvt.s.lu t3, a3, dyn
# CHECK-ASM: encoding: [0x53,0xfe,0x36,0xd0]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.s.lu t3, a3, dyn

# Rounding modes
# CHECK-ASM-AND-OBJ: fcvt.l.s a4, t4, rne
# CHECK-ASM: encoding: [0x53,0x87,0x2e,0xc0]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.l.s a4, t4, rne
# CHECK-ASM-AND-OBJ: fcvt.lu.s a5, t5, rtz
# CHECK-ASM: encoding: [0xd3,0x17,0x3f,0xc0]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.lu.s a5, t5, rtz
# CHECK-ASM-AND-OBJ: fcvt.s.l t6, a6, rdn
# CHECK-ASM: encoding: [0xd3,0x2f,0x28,0xd0]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.s.l t6, a6, rdn
# CHECK-ASM-AND-OBJ: fcvt.s.lu s7, a7, rup
# CHECK-ASM: encoding: [0xd3,0xbb,0x38,0xd0]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.s.lu s7, a7, rup
