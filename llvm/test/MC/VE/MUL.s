# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: mulu.l %s11, %s20, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x94,0x0b,0x49]
mulu.l %s11, %s20, %s22

# CHECK-INST: mulu.w %s11, 22, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x16,0x8b,0x49]
mulu.w %s11, 22, %s22

# CHECK-INST: muls.w.sx %s11, 63, (60)1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x3c,0x3f,0x0b,0x4b]
muls.w.sx %s11, 63, (60)1

# CHECK-INST: muls.w.zx %s11, -64, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x40,0x8b,0x4b]
muls.w.zx %s11, -64, %s22

# CHECK-INST: muls.l %s11, -64, (22)0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x56,0x40,0x0b,0x6e]
muls.l %s11, -64, (22)0

# CHECK-INST: muls.l.w %s11, -64, (22)0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x56,0x40,0x0b,0x6b]
muls.l.w %s11, -64, (22)0
