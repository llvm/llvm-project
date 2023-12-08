# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vfmax.d %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xbd]
vfmax.d %v11, %s20, %v22

# CHECK-INST: pvfmax.up %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x80,0xbd]
vfmax.s %vix, %vix, %vix

# CHECK-INST: pvfmax.lo %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xbd]
pvfmax.lo %vix, 22, %v22

# CHECK-INST: pvfmax.up %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0xab,0xbd]
pvfmax.up %v11, 63, %v22, %vm11

# CHECK-INST: pvfmax.up %v11, %vix, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0x0b,0x00,0x00,0x8b,0xbd]
pvfmax.up %v11, %vix, %v22, %vm11

# CHECK-INST: pvfmax %v12, %v20, %v22, %vm12
# CHECK-ENCODING: encoding: [0x00,0x16,0x14,0x0c,0x00,0x00,0xcc,0xbd]
pvfmax %v12, %v20, %v22, %vm12

# CHECK-INST: vfmin.d %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x30,0xbd]
vfmin.d %v11, %s20, %v22

# CHECK-INST: pvfmin.up %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x90,0xbd]
vfmin.s %vix, %vix, %vix

# CHECK-INST: pvfmin.lo %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x70,0xbd]
pvfmin.lo %vix, 22, %v22

# CHECK-INST: pvfmin.up %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0xbb,0xbd]
pvfmin.up %v11, 63, %v22, %vm11

# CHECK-INST: pvfmin.up %v11, %vix, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0x0b,0x00,0x00,0x9b,0xbd]
pvfmin.up %v11, %vix, %v22, %vm11

# CHECK-INST: pvfmin %v12, %v20, %v22, %vm12
# CHECK-ENCODING: encoding: [0x00,0x16,0x14,0x0c,0x00,0x00,0xdc,0xbd]
pvfmin %v12, %v20, %v22, %vm12
