# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: fidcr %s11, %s20, 0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x94,0x0b,0x51]
fidcr %s11, %s20, 0

# CHECK-INST: fidcr %s11, 22, 3
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x03,0x16,0x0b,0x51]
fidcr %s11, 22, 3

# CHECK-INST: fidcr %s11, 22, 7
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x07,0x16,0x0b,0x51]
fidcr %s11, 22, 7
