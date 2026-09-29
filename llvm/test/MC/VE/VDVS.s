# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vdivs.w.sx %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xeb]
vdivs.w.sx %v11, %s20, %v22

# CHECK-INST: vdivs.w.sx %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xeb]
vdivs.w.sx %vix, %vix, %vix

# CHECK-INST: vdivs.w.zx %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xeb]
vdivs.w.zx %vix, 22, %v22

# CHECK-INST: vdivs.w.zx %vix, %v22, 22
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x16,0x50,0xeb]
vdivs.w %vix, %v22, 22

# CHECK-INST: vdivs.w.zx %v11, %v22, 63, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x3f,0x5b,0xeb]
vdivs.w %v11, %v22, 63, %vm11
