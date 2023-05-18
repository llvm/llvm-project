# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vcmps.w.sx %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xfa]
vcmps.w.sx %v11, %s20, %v22

# CHECK-INST: vcmps.w.sx %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xfa]
vcmps.w.sx %vix, %vix, %vix

# CHECK-INST: pvcmps.lo %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xfa]
vcmps.w.zx %vix, 22, %v22

# CHECK-INST: pvcmps.lo %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xfa]
vcmps.w %vix, 22, %v22

# CHECK-INST: pvcmps.lo %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0x6b,0xfa]
pvcmps.lo %v11, 63, %v22, %vm11

# CHECK-INST: vcmps.w.sx %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0x2b,0xfa]
pvcmps.lo.sx %v11, 63, %v22, %vm11

# CHECK-INST: pvcmps.lo %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0x6b,0xfa]
pvcmps.lo.zx %v11, 63, %v22, %vm11

# CHECK-INST: pvcmps.up %v11, %vix, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0x0b,0x00,0x00,0x8b,0xfa]
pvcmps.up %v11, %vix, %v22, %vm11

# CHECK-INST: pvcmps %v12, %v20, %v22, %vm12
# CHECK-ENCODING: encoding: [0x00,0x16,0x14,0x0c,0x00,0x00,0xcc,0xfa]
pvcmps %v12, %v20, %v22, %vm12
