# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: fmax.d %s11, %s20, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x94,0x0b,0x3e]
fmax.d %s11, %s20, %s22

# CHECK-INST: fmax.s %s11, 22, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x16,0x8b,0x3e]
fmax.s %s11, 22, %s22

# CHECK-INST: fmin.d %s11, 63, (63)1
# CHECK-ENCODING: encoding: [0x80,0x00,0x00,0x00,0x3f,0x3f,0x0b,0x3e]
fmin.d %s11, 63, (63)1

# CHECK-INST: fmin.s %s11, -64, %s22
# CHECK-ENCODING: encoding: [0x80,0x00,0x00,0x00,0x96,0x40,0x8b,0x3e]
fmin.s %s11, -64, %s22
