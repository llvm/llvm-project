# RUN: llvm-mc %s -triple=riscv64 -mattr=+f -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -mattr=+f -riscv-no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -mattr=+f < %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s

# FIXME: error messages for rv32f are misleading

# CHECK-ASM-AND-OBJ: fcvt.l.s a0, ft0, dyn
# CHECK-ASM: encoding: [0x53,0x75,0x20,0xc0]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.l.s a0, ft0, dyn
# CHECK-ASM-AND-OBJ: fcvt.lu.s a1, ft1, dyn
# CHECK-ASM: encoding: [0xd3,0xf5,0x30,0xc0]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.lu.s a1, ft1, dyn
# CHECK-ASM-AND-OBJ: fcvt.s.l ft2, a2, dyn
# CHECK-ASM: encoding: [0x53,0x71,0x26,0xd0]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.s.l ft2, a2, dyn
# CHECK-ASM-AND-OBJ: fcvt.s.lu ft3, a3, dyn
# CHECK-ASM: encoding: [0xd3,0xf1,0x36,0xd0]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.s.lu ft3, a3, dyn

# Rounding modes
# CHECK-ASM-AND-OBJ: fcvt.l.s a4, ft4, rne
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.l.s a4, ft4, rne
# CHECK-ASM-AND-OBJ: fcvt.lu.s a5, ft5, rtz
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.lu.s a5, ft5, rtz
# CHECK-ASM-AND-OBJ: fcvt.s.l ft6, a6, rdn
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.s.l ft6, a6, rdn
# CHECK-ASM-AND-OBJ: fcvt.s.lu ft7, a7, rup
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.s.lu ft7, a7, rup
