# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vsubs.w.sx %v11, %s20, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xda]
vsubs.w.sx %v11, %s20, %v22

# CHECK-INST: vsubs.w.sx %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xda]
vsubs.w.sx %vix, %vix, %vix

# CHECK-INST: pvsubs.lo %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xda]
vsubs.w.zx %vix, 22, %v22

# CHECK-INST: pvsubs.lo %vix, 22, %v22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xda]
vsubs.w %vix, 22, %v22

# CHECK-INST: pvsubs.lo %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0x6b,0xda]
pvsubs.lo %v11, 63, %v22, %vm11

# CHECK-INST: vsubs.w.sx %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0x2b,0xda]
pvsubs.lo.sx %v11, 63, %v22, %vm11

# CHECK-INST: pvsubs.lo %v11, 63, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x3f,0x6b,0xda]
pvsubs.lo.zx %v11, 63, %v22, %vm11

# CHECK-INST: pvsubs.up %v11, %vix, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0x0b,0x00,0x00,0x8b,0xda]
pvsubs.up %v11, %vix, %v22, %vm11

# CHECK-INST: pvsubs %v12, %v20, %v22, %vm12
# CHECK-ENCODING: encoding: [0x00,0x16,0x14,0x0c,0x00,0x00,0xcc,0xda]
pvsubs %v12, %v20, %v22, %vm12
