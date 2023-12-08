# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: sic %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x0b,0x28]
sic %s11

# CHECK-INST: sic %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x3f,0x28]
sic %s63
