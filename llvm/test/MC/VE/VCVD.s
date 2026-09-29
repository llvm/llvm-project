# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vcvt.d.s %v11, %v12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0c,0x0b,0x00,0x00,0x00,0x8f]
vcvt.d.s %v11, %v12

# CHECK-INST: vcvt.d.s %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x0b,0x8f]
vcvt.d.s %v11, %vix, %vm11

# CHECK-INST: vcvt.d.s %vix, %v22, %vm15
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x00,0x0f,0x8f]
vcvt.d.s %vix, %v22, %vm15

# CHECK-INST: vcvt.d.s %v63, %v60, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0x3c,0x3f,0x00,0x00,0x02,0x8f]
vcvt.d.s %v63, %v60, %vm2

# CHECK-INST: vcvt.d.s %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x00,0x8f]
vcvt.d.s %vix, %vix, %vm0

# CHECK-INST: vcvt.d.s %vix, %vix, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x02,0x8f]
vcvt.d.s %vix, %vix, %vm2
