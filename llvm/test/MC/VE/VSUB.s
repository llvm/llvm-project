# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vsubu.l %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xd8]
vsubu.l %v11, %s20, %v22

# CHECK-INST: vsubu.l %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xd8]
vsubu.l %vix, %vix, %vix

# CHECK-INST: pvsubu.lo %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xd8]
vsubu.w %vix, 22, %v22

# CHECK-INST: pvsubu.lo %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0x6b,0xd8]
pvsubu.lo %v11, 63, %v22, %vm11

# CHECK-INST: pvsubu.up %v11, %vix, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0x0b,0x00,0x00,0x8b,0xd8]
pvsubu.up %v11, %vix, %v22, %vm11

# CHECK-INST: pvsubu %v12, %v20, %v22, %vm12
# CHECK-ENCODING: encoding: [0x00,0x16,0x14,0x0c,0x00,0x00,0xcc,0xd8]
pvsubu %v12, %v20, %v22, %vm12
