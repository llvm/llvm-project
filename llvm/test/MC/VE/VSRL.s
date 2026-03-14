# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vsrl %v11, %v22, %s20
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x94,0x20,0xf5]
vsrl %v11, %v22, %s20

# CHECK-INST: vsrl %vix, %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x00,0x00,0xf5]
vsrl %vix, %vix, %vix

# CHECK-INST: pvsrl.lo %vix, %v22, 22
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x00,0x16,0x60,0xf5]
pvsrl.lo %vix, %v22, 22

# CHECK-INST: pvsrl.lo %v11, %v22, 127, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x00,0x7f,0x6b,0xf5]
pvsrl.lo %v11, %v22, 127, %vm11

# CHECK-INST: pvsrl.up %v11, %vix, %v22, %vm11
# CHECK-ENCODING: encoding: [0x00,0xff,0x16,0x0b,0x00,0x00,0x8b,0xf5]
pvsrl.up %v11, %vix, %v22, %vm11

# CHECK-INST: pvsrl %v12, %v20, %v22, %vm12
# CHECK-ENCODING: encoding: [0x00,0x14,0x16,0x0c,0x00,0x00,0xcc,0xf5]
pvsrl %v12, %v20, %v22, %vm12
