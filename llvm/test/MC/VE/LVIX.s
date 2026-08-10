# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: lvix %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x8b,0x00,0xaf]
lvix %s11

# CHECK-INST: lvix 63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x3f,0x00,0xaf]
lvix 63

# CHECK-INST: lvix %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0xbf,0x00,0xaf]
lvix %s63
