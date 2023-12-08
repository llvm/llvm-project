# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vrmaxs.l.fst %v11, %v12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0c,0x0b,0x00,0x00,0x00,0xab]
vrmaxs.l.fst %v11, %v12

# CHECK-INST: vrmaxs.l.fst %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x0b,0xab]
vrmaxs.l.fst %v11, %vix, %vm11

# CHECK-INST: vrmaxs.l.lst %vix, %v22, %vm15
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x00,0x2f,0xab]
vrmaxs.l.lst %vix, %v22, %vm15

# CHECK-INST: vrmins.l.lst %v63, %v60, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0x3c,0x3f,0x00,0x00,0x32,0xab]
vrmins.l.lst %v63, %v60, %vm2

# CHECK-INST: vrmins.l.fst %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x10,0xab]
vrmins.l.fst %vix, %vix, %vm0

# CHECK-INST: vrmins.l.lst %vix, %vix, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x32,0xab]
vrmins.l.lst %vix, %vix, %vm2
