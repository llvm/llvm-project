# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

addu12i.w $a0, $a1, 1
# CHECK-INST: addu12i.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x04,0x29,0x00]

addu12i.d $a0, $a1, 1
# CHECK-INST: addu12i.d $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x84,0x29,0x00]

adc.b $a0, $a1, $a2
# CHECK-INST: adc.b $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x18,0x30,0x00]

adc.h $a0, $a1, $a2
# CHECK-INST: adc.h $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x98,0x30,0x00]

adc.w $a0, $a1, $a2
# CHECK-INST: adc.w $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x18,0x31,0x00]

adc.d $a0, $a1, $a2
# CHECK-INST: adc.d $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x98,0x31,0x00]

sbc.b $a0, $a1, $a2
# CHECK-INST: sbc.b $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x18,0x32,0x00]

sbc.h $a0, $a1, $a2
# CHECK-INST: sbc.h $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x98,0x32,0x00]

sbc.w $a0, $a1, $a2
# CHECK-INST: sbc.w $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x18,0x33,0x00]

sbc.d $a0, $a1, $a2
# CHECK-INST: sbc.d $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x98,0x33,0x00]

rotr.b $a0, $a1, $a2
# CHECK-INST: rotr.b $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x18,0x1a,0x00]

rotr.h $a0, $a1, $a2
# CHECK-INST: rotr.h $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x98,0x1a,0x00]

rotri.b $a0, $a1, 1
# CHECK-INST: rotri.b $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x24,0x4c,0x00]

rotri.h $a0, $a1, 1
# CHECK-INST: rotri.h $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x44,0x4c,0x00]

rcr.b $a0, $a1, $a2
# CHECK-INST: rcr.b $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x18,0x34,0x00]

rcr.h $a0, $a1, $a2
# CHECK-INST: rcr.h $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x98,0x34,0x00]

rcr.w $a0, $a1, $a2
# CHECK-INST: rcr.w $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x18,0x35,0x00]

rcr.d $a0, $a1, $a2
# CHECK-INST: rcr.d $a0, $a1, $a2
# CHECK-ENCODING: encoding: [0xa4,0x98,0x35,0x00]

rcri.b $a0, $a1, 1
# CHECK-INST: rcri.b $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x24,0x50,0x00]

rcri.h $a0, $a1, 1
# CHECK-INST: rcri.h $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x44,0x50,0x00]

rcri.w $a0, $a1, 1
# CHECK-INST: rcri.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x84,0x50,0x00]

rcri.d $a0, $a1, 1
# CHECK-INST: rcri.d $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x04,0x51,0x00]

fcvt.ud.d $f0, $f1
# CHECK-INST: fcvt.ud.d $fa0, $fa1
# CHECK-ENCODING: encoding: [0x20,0xe4,0x14,0x01]

fcvt.ld.d $f0, $f1
# CHECK-INST: fcvt.ld.d $fa0, $fa1
# CHECK-ENCODING: encoding: [0x20,0xe0,0x14,0x01]

fcvt.d.ld $f0, $f1, $f2
# CHECK-INST: fcvt.d.ld $fa0, $fa1, $fa2
# CHECK-ENCODING: encoding: [0x20,0x08,0x15,0x01]

ldl.d $a0, $a1, 1
# CHECK-INST: ldl.d $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x04,0x80,0x2e]

ldl.w $a0, $a1, 1
# CHECK-INST: ldl.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x04,0x00,0x2e]

ldr.w $a0, $a1, 1
# CHECK-INST: ldr.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x04,0x40,0x2e]

ldr.d $a0, $a1, 1
# CHECK-INST: ldr.d $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x04,0xc0,0x2e]

stl.w $a0, $a1, 1
# CHECK-INST: stl.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x04,0x00,0x2f]

stl.d $a0, $a1, 1
# CHECK-INST: stl.d $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x04,0x80,0x2f]

str.w $a0, $a1, 1
# CHECK-INST: str.w $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x04,0x40,0x2f]

str.d $a0, $a1, 1
# CHECK-INST: str.d $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x04,0xc0,0x2f]
