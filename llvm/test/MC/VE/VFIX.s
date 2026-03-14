# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vcvt.w.d.sx %v11, %v12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0c,0x0b,0x00,0x00,0x00,0xe8]
vcvt.w.d.sx %v11, %v12

# CHECK-INST: vcvt.w.d.zx.rz %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x08,0xff,0x0b,0x00,0x00,0x4b,0xe8]
vcvt.w.d.zx.rz %v11, %vix, %vm11

# CHECK-INST: vcvt.w.s.sx.rp %vix, %v22, %vm15
# CHECK-ENCODING: encoding: [0x00,0x09,0x16,0xff,0x00,0x00,0x8f,0xe8]
vcvt.w.s.sx.rp %vix, %v22, %vm15

# CHECK-INST: vcvt.w.s.zx.rm %v63, %v60, %vm2
# CHECK-ENCODING: encoding: [0x00,0x0a,0x3c,0x3f,0x00,0x00,0xc2,0xe8]
vcvt.w.s.zx.rm %v63, %v60, %vm2

# CHECK-INST: pvcvt.w.s.lo.rn %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x0b,0xff,0xff,0x00,0x00,0x50,0xe8]
pvcvt.w.s.lo.rn %vix, %vix, %vm0

# CHECK-INST: pvcvt.w.s.up.ra %vix, %vix, %vm2
# CHECK-ENCODING: encoding: [0x00,0x0c,0xff,0xff,0x00,0x00,0x92,0xe8]
pvcvt.w.s.up.ra %vix, %vix, %vm2

# CHECK-INST: pvcvt.w.s %vix, %vix, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0xd2,0xe8]
pvcvt.w.s %vix, %vix, %vm2
