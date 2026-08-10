# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vdivs.l %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xfb]
vdivs.l %v11, %s20, %v22

# CHECK-INST: vdivs.l %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xfb]
vdivs.l %vix, %vix, %vix

# CHECK-INST: vdivs.l %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x20,0xfb]
vdivs.l %vix, 22, %v22

# CHECK-INST: vdivs.l %v11, %v22, 63, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x3f,0x1b,0xfb]
vdivs.l %v11, %v22, 63, %vm11

# CHECK-INST: vdivs.l %v11, %v22, %s23, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x97,0x1b,0xfb]
vdivs.l %v11, %v22, %s23, %vm11
