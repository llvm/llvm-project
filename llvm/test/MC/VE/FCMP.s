# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: fcmp.d %s11, %s20, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x94,0x0b,0x7e]
fcmp.d %s11, %s20, %s22

# CHECK-INST: fcmp.s %s11, 22, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x16,0x8b,0x7e]
fcmp.s %s11, 22, %s22

# CHECK-INST: fcmp.d %s11, 63, (60)1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x3c,0x3f,0x0b,0x7e]
fcmp.d %s11, 63, (60)1

# CHECK-INST: fcmp.s %s11, -64, (22)0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x56,0x40,0x8b,0x7e]
fcmp.s %s11, -64, (22)0

# CHECK-INST: fcmp.q %s11, %s20, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x94,0x0b,0x7d]
fcmp.q %s11, %s20, %s22
