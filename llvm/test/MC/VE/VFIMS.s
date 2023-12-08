# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vfims.d %v11, %v22, %v32, 12
# CHECK-ENCODING: encoding: [0x00,0x20,0x16,0x0b,0x00,0x0c,0x00,0xff]
vfims.d %v11, %v22, %v32, 12

# CHECK-INST: vfims.d %vix, %vix, %vix, %s23
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x97,0x00,0xff]
vfims.d %vix, %vix, %vix, %s23

# CHECK-INST: vfims.s %v11, %vix, %vix, 63
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0x0b,0x00,0x3f,0x80,0xff]
vfims.s %v11, %vix, %vix, 63

# CHECK-INST: vfims.s %vix, %v20, %v12, -64
# CHECK-ENCODING: encoding: [0x00,0x0c,0x14,0xff,0x00,0x40,0x80,0xff]
vfims.s %vix, %v20, %v12, -64
