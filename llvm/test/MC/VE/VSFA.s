# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vsfa %v11, %v22, %s20, (22)0
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x56,0x94,0x00,0xd7]
vsfa %v11, %v22, %s20, (22)0

# CHECK-INST: vsfa %vix, %vix, 0, %s21
# CHECK-ENCODING: encoding: [0x00,0xff,0x00,0xff,0x95,0x00,0x00,0xd7]
vsfa %vix, %vix, 0, %s21

# CHECK-INST: vsfa %vix, %v22, 7, (22)1
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0xff,0x16,0x07,0x00,0xd7]
vsfa %vix, %v22, 7, (22)1

# CHECK-INST: vsfa %v11, %v22, %s20, %s21, %vm11
# CHECK-ENCODING: encoding: [0x00,0x16,0x00,0x0b,0x95,0x94,0x0b,0xd7]
vsfa %v11, %v22, %s20, %s21, %vm11

# CHECK-INST: vsfa %v11, %v23, %s22, (63)0, %vm11
# CHECK-ENCODING: encoding: [0x00,0x17,0x00,0x0b,0x7f,0x96,0x0b,0xd7]
vsfa %v11, %v23, %s22, (63)0, %vm11
