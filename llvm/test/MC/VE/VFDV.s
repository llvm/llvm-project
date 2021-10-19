# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vfdiv.d %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xdd]
vfdiv.d %v11, %s20, %v22

# CHECK-INST: vfdiv.d %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xdd]
vfdiv.d %vix, %vix, %vix

# CHECK-INST: vfdiv.s %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0xa0,0xdd]
vfdiv.s %vix, 22, %v22

# CHECK-INST: vfdiv.s %vix, %v22, 22
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x16,0x90,0xdd]
vfdiv.s %vix, %v22, 22

# CHECK-INST: vfdiv.s %v11, %v22, 63, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x3f,0x9b,0xdd]
vfdiv.s %v11, %v22, 63, %vm11
