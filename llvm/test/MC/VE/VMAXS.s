# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vrmaxs.w.fst.sx %v11, %v12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0c,0x0b,0x00,0x00,0x00,0xbb]
vrmaxs.w.fst.sx %v11, %v12

# CHECK-INST: vrmaxs.w.fst.sx %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x0b,0xbb]
vrmaxs.w.fst.sx %v11, %vix, %vm11

# CHECK-INST: vrmaxs.w.lst.sx %vix, %v22, %vm15
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x00,0x2f,0xbb]
vrmaxs.w.lst.sx %vix, %v22, %vm15

# CHECK-INST: vrmaxs.w.lst.zx %v63, %v60, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0x3c,0x3f,0x00,0x00,0x62,0xbb]
vrmaxs.w.lst.zx %v63, %v60, %vm2

# CHECK-INST: vrmaxs.w.fst.zx %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x40,0xbb]
vrmaxs.w.fst.zx %vix, %vix, %vm0

# CHECK-INST: vrmaxs.w.lst.zx %vix, %vix, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x62,0xbb]
vrmaxs.w.lst.zx %vix, %vix, %vm2

# CHECK-INST: vrmins.w.fst.sx %v11, %v12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0c,0x0b,0x00,0x00,0x10,0xbb]
vrmins.w.fst.sx %v11, %v12

# CHECK-INST: vrmins.w.fst.sx %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x1b,0xbb]
vrmins.w.fst.sx %v11, %vix, %vm11

# CHECK-INST: vrmins.w.lst.sx %vix, %v22, %vm15
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x00,0x3f,0xbb]
vrmins.w.lst.sx %vix, %v22, %vm15

# CHECK-INST: vrmins.w.lst.zx %v63, %v60, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0x3c,0x3f,0x00,0x00,0x72,0xbb]
vrmins.w.lst.zx %v63, %v60, %vm2

# CHECK-INST: vrmins.w.fst.zx %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x50,0xbb]
vrmins.w.fst.zx %vix, %vix, %vm0

# CHECK-INST: vrmins.w.lst.zx %vix, %vix, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x72,0xbb]
vrmins.w.lst.zx %vix, %vix, %vm2
