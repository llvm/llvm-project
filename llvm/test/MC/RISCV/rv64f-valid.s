# RUN: llvm-mc %s -triple=riscv64 -mattr=+f -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump -mattr=+f -d - | FileCheck -check-prefix=CHECK-INST %s
# RUN: not llvm-mc -triple riscv32 -mattr=+f < %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s

# CHECK-INST: fcvt.l.s a0, ft0
# CHECK: encoding: [0x53,0x75,0x20,0xc0]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.l.s a0, ft0
# CHECK-INST: fcvt.lu.s a1, ft1
# CHECK: encoding: [0xd3,0xf5,0x30,0xc0]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.lu.s a1, ft1
# CHECK-INST: fcvt.s.l ft2, a2
# CHECK: encoding: [0x53,0x71,0x26,0xd0]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.s.l ft2, a2
# CHECK-INST: fcvt.s.lu ft3, a3
# CHECK: encoding: [0xd3,0xf1,0x36,0xd0]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.s.lu ft3, a3

# Rounding modes
# CHECK-INST: fcvt.l.s a4, ft4, rne
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.l.s a4, ft4, rne
# CHECK-INST: fcvt.lu.s a5, ft5, rtz
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.lu.s a5, ft5, rtz
# CHECK-INST: fcvt.s.l ft6, a6, rdn
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.s.l ft6, a6, rdn
# CHECK-INST: fcvt.s.lu ft7, a7, rup
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.s.lu ft7, a7, rup
