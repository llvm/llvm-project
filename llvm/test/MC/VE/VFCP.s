# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vfcmp.d %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xfc]
vfcmp.d %v11, %s20, %v22

# CHECK-INST: pvfcmp.up %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x80,0xfc]
vfcmp.s %vix, %vix, %vix

# CHECK-INST: pvfcmp.lo %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xfc]
pvfcmp.lo %vix, 22, %v22

# CHECK-INST: pvfcmp.up %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0xab,0xfc]
pvfcmp.up %v11, 63, %v22, %vm11

# CHECK-INST: pvfcmp.up %v11, %vix, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0x0b,0x00,0x00,0x8b,0xfc]
pvfcmp.up %v11, %vix, %v22, %vm11

# CHECK-INST: pvfcmp %v12, %v20, %v22, %vm12
# CHECK-ENCODING: encoding: [0x00,0x16,0x14,0x0c,0x00,0x00,0xcc,0xfc]
pvfcmp %v12, %v20, %v22, %vm12
