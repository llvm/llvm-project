# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vfnmsb.d %v11, %s20, %v22, %v21
# CHECK-ENCODING: encoding: [0x15,0x16,0x00,0x0b,0x00,0x94,0x20,0xf3]
vfnmsb.d %v11, %s20, %v22, %v21

# CHECK-INST: pvfnmsb.up %vix, %vix, %vix, %v21
# CHECK-ENCODING: encoding: [0x15,0xff,0xff,0xff,0x00,0x00,0x80,0xf3]
vfnmsb.s %vix, %vix, %vix, %v21

# CHECK-INST: pvfnmsb.lo %vix, 22, %v22, %vix
# CHECK-ENCODING: encoding: [0xff,0x16,0x00,0xff,0x00,0x16,0x60,0xf3]
pvfnmsb.lo %vix, 22, %v22, %vix

# CHECK-INST: pvfnmsb.up %vix, %v22, 22, %vix
# CHECK-ENCODING: encoding: [0xff,0x00,0x16,0xff,0x00,0x16,0x90,0xf3]
pvfnmsb.up %vix, %v22, 22, %vix

# CHECK-INST: pvfnmsb %v11, %v22, 63, %v20, %vm12
# CHECK-ENCODING: encoding: [0x14,0x00,0x16,0x0b,0x00,0x3f,0xdc,0xf3]
pvfnmsb %v11, %v22, 63, %v20, %vm12
