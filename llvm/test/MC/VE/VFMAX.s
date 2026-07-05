# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vfrmax.d.fst %v11, %v12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0c,0x0b,0x00,0x00,0x00,0xad]
vfrmax.d.fst %v11, %v12

# CHECK-INST: vfrmax.d.fst %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x0b,0xad]
vfrmax.d.fst %v11, %vix, %vm11

# CHECK-INST: vfrmax.d.lst %vix, %v22, %vm15
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x00,0x2f,0xad]
vfrmax.d.lst %vix, %v22, %vm15

# CHECK-INST: vfrmax.s.lst %v63, %v60, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0x3c,0x3f,0x00,0x00,0xa2,0xad]
vfrmax.s.lst %v63, %v60, %vm2

# CHECK-INST: vfrmax.s.fst %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x80,0xad]
vfrmax.s.fst %vix, %vix, %vm0

# CHECK-INST: vfrmax.s.lst %vix, %vix, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0xa2,0xad]
vfrmax.s.lst %vix, %vix, %vm2

# CHECK-INST: vfrmin.d.fst %v11, %v12
# CHECK-ENCODING: encoding: [0x00,0x00,0x0c,0x0b,0x00,0x00,0x10,0xad]
vfrmin.d.fst %v11, %v12

# CHECK-INST: vfrmin.d.fst %v11, %vix, %vm11
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0x0b,0x00,0x00,0x1b,0xad]
vfrmin.d.fst %v11, %vix, %vm11

# CHECK-INST: vfrmin.d.lst %vix, %v22, %vm15
# CHECK-ENCODING: encoding: [0x00,0x00,0x16,0xff,0x00,0x00,0x3f,0xad]
vfrmin.d.lst %vix, %v22, %vm15

# CHECK-INST: vfrmin.s.lst %v63, %v60, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0x3c,0x3f,0x00,0x00,0xb2,0xad]
vfrmin.s.lst %v63, %v60, %vm2

# CHECK-INST: vfrmin.s.fst %vix, %vix
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0x90,0xad]
vfrmin.s.fst %vix, %vix, %vm0

# CHECK-INST: vfrmin.s.lst %vix, %vix, %vm2
# CHECK-ENCODING: encoding: [0x00,0x00,0xff,0xff,0x00,0x00,0xb2,0xad]
vfrmin.s.lst %vix, %vix, %vm2
