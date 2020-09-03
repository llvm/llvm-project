# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: sra.w.sx %s11, %s11, %s11
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x8b,0x0b,0x76]
sra.w.sx %s11, %s11, %s11

# CHECK-INST: sra.w.zx %s11, %s11, 63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x3f,0x8b,0x76]
sra.w.zx %s11, %s11, 63

# CHECK-INST: sra.l %s11, %s11, 127
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x7f,0x0b,0x77]
sra.l %s11, %s11, 127

# CHECK-INST: sra.w.sx %s11, %s11, 64
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x8b,0x40,0x0b,0x76]
sra.w.sx %s11, %s11, 64

# CHECK-INST: sra.w.zx %s11, (32)1, 64
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x20,0x40,0x8b,0x76]
sra.w.zx %s11, (32)1, 64

# CHECK-INST: sra.l %s11, (32)0, 63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x60,0x3f,0x0b,0x77]
sra.l %s11, (32)0, 63
