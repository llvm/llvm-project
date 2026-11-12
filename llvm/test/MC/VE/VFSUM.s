# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vfsum.d %v11, %v12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0c,0x0b,0x00,0x00,0x00,0xec]
vfsum.d %v11, %v12

# CHECK-INST: vfsum.d %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x0b,0xec]
vfsum.d %v11, %vix, %vm11

# CHECK-INST: vfsum.d %vix, %v22, %vm15
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x00,0x0f,0xec]
vfsum.d %vix, %v22, %vm15

# CHECK-INST: vfsum.s %v63, %v60, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0x3c,0x3f,0x00,0x00,0x82,0xec]
vfsum.s %v63, %v60, %vm2

# CHECK-INST: vfsum.s %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x80,0xec]
vfsum.s %vix, %vix, %vm0

# CHECK-INST: vfsum.s %vix, %vix, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x82,0xec]
vfsum.s %vix, %vix, %vm2
