# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vbitclr.b $vr1, $vr0, $vr30
# CHECK-INST: vbitclr.b $vr1, $vr0, $vr30
# CHECK-ENCODING: encoding: [0x01,0x78,0x0c,0x71]

vbitclr.h $vr27, $vr5, $vr28
# CHECK-INST: vbitclr.h $vr27, $vr5, $vr28
# CHECK-ENCODING: encoding: [0xbb,0xf0,0x0c,0x71]

vbitclr.w $vr3, $vr30, $vr14
# CHECK-INST: vbitclr.w $vr3, $vr30, $vr14
# CHECK-ENCODING: encoding: [0xc3,0x3b,0x0d,0x71]

vbitclr.d $vr25, $vr11, $vr4
# CHECK-INST: vbitclr.d $vr25, $vr11, $vr4
# CHECK-ENCODING: encoding: [0x79,0x91,0x0d,0x71]

vbitclri.b $vr15, $vr25, 4
# CHECK-INST: vbitclri.b $vr15, $vr25, 4
# CHECK-ENCODING: encoding: [0x2f,0x33,0x10,0x73]

vbitclri.h $vr24, $vr22, 1
# CHECK-INST: vbitclri.h $vr24, $vr22, 1
# CHECK-ENCODING: encoding: [0xd8,0x46,0x10,0x73]

vbitclri.w $vr30, $vr20, 1
# CHECK-INST: vbitclri.w $vr30, $vr20, 1
# CHECK-ENCODING: encoding: [0x9e,0x86,0x10,0x73]

vbitclri.d $vr5, $vr0, 16
# CHECK-INST: vbitclri.d $vr5, $vr0, 16
# CHECK-ENCODING: encoding: [0x05,0x40,0x11,0x73]
