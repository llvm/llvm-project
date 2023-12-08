# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: vmv %v11, 23, %v11
# CHECK-ENCODING: encoding: [0x00,0x0b,0x00,0x0b,0x00,0x17,0x00,0x9c]
vmv %v11, 23, %v11

# CHECK-INST: vmv %v11, %s12, %vix, %vm15
# CHECK-ENCODING: encoding: [0x00,0xff,0x00,0x0b,0x00,0x8c,0x0f,0x9c]
vmv %v11, %s12, %vix, %vm15

# CHECK-INST: vmv %vix, 127, %v63
# CHECK-ENCODING: encoding: [0x00,0x3f,0x00,0xff,0x00,0x7f,0x00,0x9c]
vmv %vix, 127, %v63, %vm0
