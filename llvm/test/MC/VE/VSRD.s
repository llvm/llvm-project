# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vsrd %v11, (%v22, %v23), %s20
# CHECK-ENCODING: encoding: [0x00,0x17,0x16,0x0b,0x00,0x94,0x00,0xf4]
vsrd %v11, (%v22, %v23), %s20

# CHECK-INST: vsrd %vix, (%vix, %vix), %s23
# CHECK-ENCODING: encoding: [0x00,0xff,0xff,0xff,0x00,0x97,0x00,0xf4]
vsrd %vix, (%vix, %vix), %s23

# CHECK-INST: vsrd %vix, (%v22, %v30), 22
# CHECK-ENCODING: encoding: [0x00,0x1e,0x16,0xff,0x00,0x16,0x00,0xf4]
vsrd %vix, (%v22, %v30), 22

# CHECK-INST: vsrd %v11, (%v22, %vix), 127, %vm11
# CHECK-ENCODING: encoding: [0x00,0xff,0x16,0x0b,0x00,0x7f,0x0b,0xf4]
vsrd %v11, (%v22, %vix), 127, %vm11

# CHECK-INST: vsrd %v11, (%vix, %v22), 21, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0xff,0x0b,0x00,0x15,0x0b,0xf4]
vsrd %v11, (%vix, %v22), 21, %vm11

# CHECK-INST: vsrd %v12, (%v20, %v22), 2, %vm12
# CHECK-ENCODING: encoding: [0x00,0x16,0x14,0x0c,0x00,0x02,0x0c,0xf4]
vsrd %v12, (%v20, %v22), 2, %vm12
