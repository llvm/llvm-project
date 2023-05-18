# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vfima.d %v11, %v22, %v32, 12
# CHECK-ENCODING: encoding: [0x00,0x20,0x16,0x0b,0x00,0x0c,0x00,0xef]
vfima.d %v11, %v22, %v32, 12

# CHECK-INST: vfima.d %vix, %vix, %vix, %s23
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x97,0x00,0xef]
vfima.d %vix, %vix, %vix, %s23

# CHECK-INST: vfima.s %v11, %vix, %vix, 63
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0x0b,0x00,0x3f,0x80,0xef]
vfima.s %v11, %vix, %vix, 63

# CHECK-INST: vfima.s %vix, %v20, %v12, -64
# CHECK-ENCODING: encoding: [0x00,0x0c,0x14,0xff,0x00,0x40,0x80,0xef]
vfima.s %vix, %v20, %v12, -64
