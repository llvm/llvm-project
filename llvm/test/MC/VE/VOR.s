# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vor %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xc5]
vor %v11, %s20, %v22

# CHECK-INST: vor %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xc5]
vor %vix, %vix, %vix

# CHECK-INST: pvor.lo %vix, (22)1, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xc5]
pvor.lo %vix, (22)1, %v22

# CHECK-INST: pvor.lo %v11, (63)0, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x7f,0x6b,0xc5]
pvor.lo %v11, (63)0, %v22, %vm11

# CHECK-INST: pvor.up %v11, %vix, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0x0b,0x00,0x00,0x8b,0xc5]
pvor.up %v11, %vix, %v22, %vm11

# CHECK-INST: pvor %v12, %v20, %v22, %vm12
# CHECK-ENCODING: encoding: [0x00,0x16,0x14,0x0c,0x00,0x00,0xcc,0xc5]
pvor %v12, %v20, %v22, %vm12
