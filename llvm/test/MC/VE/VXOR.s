# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vxor %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xc6]
vxor %v11, %s20, %v22

# CHECK-INST: vxor %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xc6]
vxor %vix, %vix, %vix

# CHECK-INST: pvxor.lo %vix, (22)0, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x56,0x60,0xc6]
pvxor.lo %vix, (22)0, %v22

# CHECK-INST: pvxor.lo %v11, (63)1, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0x6b,0xc6]
pvxor.lo %v11, (63)1, %v22, %vm11

# CHECK-INST: pvxor.up %v11, %vix, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0x0b,0x00,0x00,0x8b,0xc6]
pvxor.up %v11, %vix, %v22, %vm11

# CHECK-INST: pvxor %v12, %v20, %v22, %vm12
# CHECK-ENCODING: encoding: [0x00,0x16,0x14,0x0c,0x00,0x00,0xcc,0xc6]
pvxor %v12, %v20, %v22, %vm12
