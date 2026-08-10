# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: cvt.q.d %s18, %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x8c,0x12,0x2d]
cvt.q.d %s18, %s12

# CHECK-INST: cvt.q.d %s18, 63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x3f,0x12,0x2d]
cvt.q.d %s18, 63

# CHECK-INST: cvt.q.d %s18, -64
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x40,0x12,0x2d]
cvt.q.d %s18, -64

# CHECK-INST: cvt.q.d %s18, -1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x7f,0x12,0x2d]
cvt.q.d %s18, -1
