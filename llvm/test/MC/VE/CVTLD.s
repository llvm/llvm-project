# RUN: llvm-mc -triple=ve --show-encoding < %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=ve -filetype=obj < %s | llvm-objdump -d - \
# RUN:     | FileCheck %s --check-prefixes=CHECK-INST

# CHECK-INST: cvt.l.d %s11, %s12
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x00,0x8c,0x0b,0x4f]
cvt.l.d %s11, %s12

# CHECK-INST: cvt.l.d.rz %s11, 63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x08,0x3f,0x0b,0x4f]
cvt.l.d.rz %s11, 63

# CHECK-INST: cvt.l.d.rp %s11, -64
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x09,0x40,0x0b,0x4f]
cvt.l.d.rp %s11, -64

# CHECK-INST: cvt.l.d.rm %s11, -1
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x0a,0x7f,0x0b,0x4f]
cvt.l.d.rm %s11, -1

# CHECK-INST: cvt.l.d.rn %s11, 7
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x0b,0x07,0x0b,0x4f]
cvt.l.d.rn %s11, 7

# CHECK-INST: cvt.l.d.ra %s11, %s63
# CHECK-ENCODING: encoding: [0x00,0x00,0x00,0x00,0x0c,0xbf,0x0b,0x4f]
cvt.l.d.ra %s11, %s63
