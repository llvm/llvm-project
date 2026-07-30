# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: subu.l %s11, %s20, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x94,0x0b,0x58]
subu.l %s11, %s20, %s22

# CHECK-INST: subu.w %s11, 22, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x16,0x8b,0x58]
subu.w %s11, 22, %s22

# CHECK-INST: subs.w.sx %s11, 63, (60)1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x3c,0x3f,0x0b,0x5a]
subs.w.sx %s11, 63, (60)1

# CHECK-INST: subs.w.zx %s11, -64, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x40,0x8b,0x5a]
subs.w.zx %s11, -64, %s22

# CHECK-INST: subs.l %s11, -64, (22)0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x56,0x40,0x0b,0x5b]
subs.l %s11, -64, (22)0
