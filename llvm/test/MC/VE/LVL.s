# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: lvl %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x8b,0x00,0xbf]
lvl %s11

# CHECK-INST: lvl 63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x3f,0x00,0xbf]
lvl 63

# CHECK-INST: lvl -64
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x40,0x00,0xbf]
lvl -64
