# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: ts2am %s20, 20(%s11), %s32
# CHECK-ENCODING: encoding: [0x14,0x00,0x00,0x00,0x8b,0xa0,0x14,0x43]
ts2am %s20, 20(%s11), %s32

# CHECK-INST: ts2am %s20, 8192, 127
# CHECK-ENCODING: encoding: [0x00,0x20,0x00,0x00,0x00,0x7f,0x14,0x43]
ts2am %s20, 8192, 127
