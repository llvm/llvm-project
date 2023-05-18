# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vcvt.d.w %v11, %v12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0c,0x0b,0x00,0x00,0x00,0xf8]
vcvt.d.w %v11, %v12

# CHECK-INST: vcvt.s.w %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x8b,0xf8]
vcvt.s.w %v11, %vix, %vm11

# CHECK-INST: pvcvt.s.w.lo %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x50,0xf8]
pvcvt.s.w.lo %vix, %vix, %vm0

# CHECK-INST: pvcvt.s.w.up %vix, %vix, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x92,0xf8]
pvcvt.s.w.up %vix, %vix, %vm2

# CHECK-INST: pvcvt.s.w %vix, %vix, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0xd2,0xf8]
pvcvt.s.w %vix, %vix, %vm2
