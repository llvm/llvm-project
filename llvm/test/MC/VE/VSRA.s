# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vsra.w.sx %v11, %v22, %s20
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xf6]
vsra.w.sx %v11, %v22, %s20

# CHECK-INST: vsra.w.sx %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xf6]
vsra.w.sx %vix, %vix, %vix

# CHECK-INST: vsra.w.sx %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xf6]
pvsra.lo.sx %vix, %vix, %vix

# CHECK-INST: pvsra.lo %vix, %v22, 22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xf6]
pvsra.lo %vix, %v22, 22

# CHECK-INST: pvsra.lo %vix, %v22, 22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xf6]
vsra.w.zx %vix, %v22, 22

# CHECK-INST: pvsra.lo %vix, %v22, 22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xf6]
vsra.w %vix, %v22, 22

# CHECK-INST: pvsra.lo %v11, %v22, 127, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x7f,0x6b,0xf6]
pvsra.lo.zx %v11, %v22, 127, %vm11

# CHECK-INST: pvsra.up %v11, %vix, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0xff,0x16,0x0b,0x00,0x00,0x8b,0xf6]
pvsra.up %v11, %vix, %v22, %vm11

# CHECK-INST: pvsra %v12, %v20, %v22, %vm12
# CHECK-ENCODING: encoding: [0x00,0x14,0x16,0x0c,0x00,0x00,0xcc,0xf6]
pvsra %v12, %v20, %v22, %vm12
