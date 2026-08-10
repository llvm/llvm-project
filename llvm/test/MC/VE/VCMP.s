# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vcmpu.l %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xb9]
vcmpu.l %v11, %s20, %v22

# CHECK-INST: vcmpu.l %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xb9]
vcmpu.l %vix, %vix, %vix

# CHECK-INST: pvcmpu.lo %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xb9]
vcmpu.w %vix, 22, %v22

# CHECK-INST: pvcmpu.lo %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0x6b,0xb9]
pvcmpu.lo %v11, 63, %v22, %vm11

# CHECK-INST: pvcmpu.up %v11, %vix, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0x0b,0x00,0x00,0x8b,0xb9]
pvcmpu.up %v11, %vix, %v22, %vm11

# CHECK-INST: pvcmpu %v12, %v20, %v22, %vm12
# CHECK-ENCODING: encoding: [0x00,0x16,0x14,0x0c,0x00,0x00,0xcc,0xb9]
pvcmpu %v12, %v20, %v22, %vm12
