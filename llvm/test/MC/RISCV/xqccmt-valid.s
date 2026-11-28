# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqccmt \
# RUN:   -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-xqccmt < %s \
# RUN:     | llvm-objdump --mattr=-c,+experimental-xqccmt --no-print-imm-hex \
# RUN:   -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-xqccmt \
# RUN:   -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-xqccmt < %s \
# RUN:     | llvm-objdump --mattr=-c,+experimental-xqccmt --no-print-imm-hex \
# RUN:   -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple riscv64 -M no-aliases -show-encoding < %s

# qc.cm.jt
# CHECK-ASM-AND-OBJ: qc.cm.jt 1
# CHECK-ASM: encoding: [0x06,0xa0]
qc.cm.jt 1 # expected-error {{instruction requires the following: 'Xqccmt' (Qualcomm 16-bit Table Jump)}}

# CHECK-ASM-AND-OBJ: qc.cm.jt 0
# CHECK-ASM: encoding: [0x02,0xa0]
qc.cm.jt 0 # expected-error {{instruction requires the following: 'Xqccmt' (Qualcomm 16-bit Table Jump)}}

# CHECK-ASM-AND-OBJ: qc.cm.jt 31
# CHECK-ASM: encoding: [0x7e,0xa0]
qc.cm.jt 31 # expected-error {{instruction requires the following: 'Xqccmt' (Qualcomm 16-bit Table Jump)}}

# qc.cm.jalt
# CHECK-ASM-AND-OBJ: qc.cm.jalt 32
# CHECK-ASM: encoding: [0x82,0xa0]
qc.cm.jalt 32 # expected-error {{instruction requires the following: 'Xqccmt' (Qualcomm 16-bit Table Jump)}}

# CHECK-ASM-AND-OBJ: qc.cm.jalt 255
# CHECK-ASM: encoding: [0xfe,0xa3]
qc.cm.jalt 255 # expected-error {{instruction requires the following: 'Xqccmt' (Qualcomm 16-bit Table Jump)}}
