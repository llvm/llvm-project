# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: smvl %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x0b,0x2e]
smvl %s11

# CHECK-INST: smvl %s0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x2e]
smvl %s0

# CHECK-INST: smvl %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x3f,0x2e]
smvl %s63
