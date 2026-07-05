# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vfmad.d %v11, %s20, %v22, %v21
# CHECK-ENCODING: encoding: [0x15,0x16,0x00,0x0b,0x00,0x94,0x20,0xe2]
vfmad.d %v11, %s20, %v22, %v21

# CHECK-INST: pvfmad.up %vix, %vix, %vix, %v21
# CHECK-ENCODING: encoding: [0x15,0xff,0xff,0xff,0x00,0x00,0x80,0xe2]
vfmad.s %vix, %vix, %vix, %v21

# CHECK-INST: pvfmad.lo %vix, 22, %v22, %vix
# CHECK-ENCODING: encoding: [0xff,0x16,0x00,0xff,0x00,0x16,0x60,0xe2]
pvfmad.lo %vix, 22, %v22, %vix

# CHECK-INST: pvfmad.up %vix, %v22, 22, %vix
# CHECK-ENCODING: encoding: [0xff,0x00,0x16,0xff,0x00,0x16,0x90,0xe2]
pvfmad.up %vix, %v22, 22, %vix

# CHECK-INST: pvfmad %v11, %v22, 63, %v20, %vm12
# CHECK-ENCODING: encoding: [0x14,0x00,0x16,0x0b,0x00,0x3f,0xdc,0xe2]
pvfmad %v11, %v22, 63, %v20, %vm12
