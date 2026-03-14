# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vfsqrt.d %v11, %v22
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x00,0x00,0xed]
vfsqrt.d %v11, %v22

# CHECK-INST: vfsqrt.d %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x00,0xed]
vfsqrt.d %vix, %vix

# CHECK-INST: vfsqrt.s %vix, %v22
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x00,0x80,0xed]
vfsqrt.s %vix, %v22

# CHECK-INST: vfsqrt.s %v11, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0x0b,0x00,0x00,0x8b,0xed]
vfsqrt.s %v11, %v22, %vm11

# CHECK-INST: vfsqrt.s %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x8b,0xed]
vfsqrt.s %v11, %vix, %vm11

# CHECK-INST: vfsqrt.s %v12, %v20, %vm12
# CHECK-ENCODING: encoding: [0x00,0x00,0x14,0x0c,0x00,0x00,0x8c,0xed]
vfsqrt.s %v12, %v20, %vm12
