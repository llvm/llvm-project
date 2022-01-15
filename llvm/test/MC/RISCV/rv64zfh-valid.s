# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfh -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zfh < %s \
# RUN:     | llvm-objdump --mattr=+zfh -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -mattr=+zfh < %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s

# CHECK-ASM-AND-OBJ: fcvt.l.h a0, ft0, dyn
# CHECK-ASM: encoding: [0x53,0x75,0x20,0xc4]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.l.h a0, ft0, dyn
# CHECK-ASM-AND-OBJ: fcvt.lu.h a1, ft1, dyn
# CHECK-ASM: encoding: [0xd3,0xf5,0x30,0xc4]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.lu.h a1, ft1, dyn
# CHECK-ASM-AND-OBJ: fcvt.h.l ft2, a2, dyn
# CHECK-ASM: encoding: [0x53,0x71,0x26,0xd4]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.h.l ft2, a2, dyn
# CHECK-ASM-AND-OBJ: fcvt.h.lu ft3, a3, dyn
# CHECK-ASM: encoding: [0xd3,0xf1,0x36,0xd4]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.h.lu ft3, a3, dyn

# Rounding modes
# CHECK-ASM-AND-OBJ: fcvt.l.h a4, ft4, rne
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.l.h a4, ft4, rne
# CHECK-ASM-AND-OBJ: fcvt.lu.h a5, ft5, rtz
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.lu.h a5, ft5, rtz
# CHECK-ASM-AND-OBJ: fcvt.h.l ft6, a6, rdn
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.h.l ft6, a6, rdn
# CHECK-ASM-AND-OBJ: fcvt.h.lu ft7, a7, rup
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
fcvt.h.lu ft7, a7, rup
