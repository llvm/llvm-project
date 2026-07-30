# RUN: llvm-mc %s -triple=riscv64 -mattr=+zdinx -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zdinx %s \
# RUN:     | llvm-objdump --mattr=+zdinx -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -mattr=+zdinx %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s

# CHECK-ASM-AND-OBJ: fcvt.l.d a0, t0, dyn
# CHECK-ASM: encoding: [0x53,0xf5,0x22,0xc2]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.l.d a0, t0, dyn
# CHECK-ASM-AND-OBJ: fcvt.lu.d a1, t1, dyn
# CHECK-ASM: encoding: [0xd3,0x75,0x33,0xc2]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.lu.d a1, t1, dyn
# CHECK-ASM-AND-OBJ: fcvt.d.l t3, a3, dyn
# CHECK-ASM: encoding: [0x53,0xfe,0x26,0xd2]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.d.l t3, a3, dyn
# CHECK-ASM-AND-OBJ: fcvt.d.lu t4, a4, dyn
# CHECK-ASM: encoding: [0xd3,0x7e,0x37,0xd2]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.d.lu t4, a4, dyn

# Rounding modes
# CHECK-ASM-AND-OBJ: fcvt.d.l t3, a3, rne
# CHECK-ASM: encoding: [0x53,0x8e,0x26,0xd2]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.d.l t3, a3, rne
# CHECK-ASM-AND-OBJ: fcvt.d.lu t4, a4, rtz
# CHECK-ASM: encoding: [0xd3,0x1e,0x37,0xd2]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.d.lu t4, a4, rtz
# CHECK-ASM-AND-OBJ: fcvt.l.d a0, t0, rdn
# CHECK-ASM: encoding: [0x53,0xa5,0x22,0xc2]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.l.d a0, t0, rdn
# CHECK-ASM-AND-OBJ: fcvt.lu.d a1, t1, rup
# CHECK-ASM: encoding: [0xd3,0x35,0x33,0xc2]
# CHECK-RV32: :[[#@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.lu.d a1, t1, rup
