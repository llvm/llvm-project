# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: maxs.w.sx %s11, %s20, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x94,0x0b,0x78]
maxs.w.sx %s11, %s20, %s22

# CHECK-INST: maxs.w.zx %s11, 22, %s22
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x96,0x16,0x8b,0x78]
maxs.w.zx %s11, 22, %s22

# CHECK-INST: mins.w.sx %s11, 63, (63)1
# CHECK-ENCODING: encoding: [0x80,0x00,0x00,0x00,0x3f,0x3f,0x0b,0x78]
mins.w.sx %s11, 63, (63)1

# CHECK-INST: mins.w.zx %s11, -64, %s22
# CHECK-ENCODING: encoding: [0x80,0x00,0x00,0x00,0x96,0x40,0x8b,0x78]
mins.w.zx %s11, -64, %s22

# CHECK-INST: maxs.l %s11, -64, (22)0
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x56,0x40,0x0b,0x68]
maxs.l %s11, -64, (22)0

# CHECK-INST: mins.l %s11, -64, (22)1
# CHECK-ENCODING: encoding: [0x80,0x00,0x00,0x00,0x16,0x40,0x0b,0x68]
mins.l %s11, -64, (22)1
