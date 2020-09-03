# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: lfr %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x8b,0x00,0x69]
lfr %s11

# CHECK-INST: lfr 63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x3f,0x00,0x69]
lfr 63

# CHECK-INST: sfr %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x3f,0x29]
sfr %s63
