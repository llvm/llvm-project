# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: lpm %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x8b,0x00,0x3a]
lpm %s11

# CHECK-INST: spm %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x3f,0x2a]
spm %s63
