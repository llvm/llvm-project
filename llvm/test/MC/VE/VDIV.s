# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vdivu.l %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xe9]
vdivu.l %v11, %s20, %v22

# CHECK-INST: vdivu.l %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xe9]
vdivu.l %vix, %vix, %vix

# CHECK-INST: vdivu.w %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xe9]
vdivu.w %vix, 22, %v22

# CHECK-INST: vdivu.l %v11, %v22, 63, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x3f,0x1b,0xe9]
vdivu.l %v11, %v22, 63, %vm11

# CHECK-INST: vdivu.l %v11, %vix, %s22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x96,0x1b,0xe9]
vdivu.l %v11, %vix, %s22, %vm11
