# RUN: llvm-mc %s -triple=riscv64 -mattr=+d -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+d < %s \
# RUN:     | llvm-objdump -mattr=+d -d - | FileCheck -check-prefix=CHECK-INST %s
# RUN: not llvm-mc -triple riscv32 -mattr=+d < %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s

# CHECK-INST: fcvt.l.d a0, ft0
# CHECK: encoding: [0x53,0x75,0x20,0xc2]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.l.d a0, ft0
# CHECK-INST: fcvt.lu.d a1, ft1
# CHECK: encoding: [0xd3,0xf5,0x30,0xc2]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.lu.d a1, ft1
# CHECK-INST: fmv.x.d a2, ft2
# CHECK: encoding: [0x53,0x06,0x01,0xe2]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fmv.x.d a2, ft2
# CHECK-INST: fcvt.d.l ft3, a3
# CHECK: encoding: [0xd3,0xf1,0x26,0xd2]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.d.l ft3, a3
# CHECK-INST: fcvt.d.lu ft4, a4
# CHECK: encoding: [0x53,0x72,0x37,0xd2]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.d.lu ft4, a4
# CHECK-INST: fmv.d.x ft5, a5
# CHECK: encoding: [0xd3,0x82,0x07,0xf2]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fmv.d.x ft5, a5

# Rounding modes
# CHECK-INST: fcvt.d.l ft3, a3, rne
# CHECK: encoding: [0xd3,0x81,0x26,0xd2]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.d.l ft3, a3, rne
# CHECK-INST: fcvt.d.lu ft4, a4, rtz
# CHECK: encoding: [0x53,0x12,0x37,0xd2]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.d.lu ft4, a4, rtz
# CHECK-INST: fcvt.l.d a0, ft0, rdn
# CHECK: encoding: [0x53,0x25,0x20,0xc2]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.l.d a0, ft0, rdn
# CHECK-INST: fcvt.lu.d a1, ft1, rup
# CHECK: encoding: [0xd3,0xb5,0x30,0xc2]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction use requires an option to be enabled
fcvt.lu.d a1, ft1, rup
