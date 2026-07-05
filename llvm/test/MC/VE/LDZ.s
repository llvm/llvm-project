# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: ldz %s11, %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x00,0x0b,0x67]
ldz %s11, %s11

# CHECK-INST: ldz %s11, (32)1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x20,0x00,0x0b,0x67]
ldz %s11, (32)1

# CHECK-INST: ldz %s11, (63)0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x7f,0x00,0x0b,0x67]
ldz %s11, (63)0
