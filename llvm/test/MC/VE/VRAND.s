# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vrand %v11, %v22
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x00,0x00,0x88]
vrand %v11, %v22

# CHECK-INST: vrand %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x00,0x88]
vrand %vix, %vix

# CHECK-INST: vrand %v11, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x00,0x0b,0x88]
vrand %v11, %v22, %vm11

# CHECK-INST: vrand %v11, %vix, %vm15
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x0f,0x88]
vrand %v11, %vix, %vm15
