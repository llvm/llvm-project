# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vaddwev.h.b $vr2, $vr23, $vr25
# CHECK-INST: vaddwev.h.b $vr2, $vr23, $vr25
# CHECK-ENCODING: encoding: [0xe2,0x66,0x1e,0x70]

vaddwev.w.h $vr4, $vr8, $vr30
# CHECK-INST: vaddwev.w.h $vr4, $vr8, $vr30
# CHECK-ENCODING: encoding: [0x04,0xf9,0x1e,0x70]

vaddwev.d.w $vr8, $vr31, $vr5
# CHECK-INST: vaddwev.d.w $vr8, $vr31, $vr5
# CHECK-ENCODING: encoding: [0xe8,0x17,0x1f,0x70]

vaddwev.q.d $vr10, $vr10, $vr13
# CHECK-INST: vaddwev.q.d $vr10, $vr10, $vr13
# CHECK-ENCODING: encoding: [0x4a,0xb5,0x1f,0x70]

vaddwev.h.bu $vr12, $vr24, $vr25
# CHECK-INST: vaddwev.h.bu $vr12, $vr24, $vr25
# CHECK-ENCODING: encoding: [0x0c,0x67,0x2e,0x70]

vaddwev.w.hu $vr3, $vr9, $vr30
# CHECK-INST: vaddwev.w.hu $vr3, $vr9, $vr30
# CHECK-ENCODING: encoding: [0x23,0xf9,0x2e,0x70]

vaddwev.d.wu $vr27, $vr10, $vr17
# CHECK-INST: vaddwev.d.wu $vr27, $vr10, $vr17
# CHECK-ENCODING: encoding: [0x5b,0x45,0x2f,0x70]

vaddwev.q.du $vr25, $vr20, $vr14
# CHECK-INST: vaddwev.q.du $vr25, $vr20, $vr14
# CHECK-ENCODING: encoding: [0x99,0xba,0x2f,0x70]

vaddwev.h.bu.b $vr5, $vr7, $vr16
# CHECK-INST: vaddwev.h.bu.b $vr5, $vr7, $vr16
# CHECK-ENCODING: encoding: [0xe5,0x40,0x3e,0x70]

vaddwev.w.hu.h $vr15, $vr13, $vr29
# CHECK-INST: vaddwev.w.hu.h $vr15, $vr13, $vr29
# CHECK-ENCODING: encoding: [0xaf,0xf5,0x3e,0x70]

vaddwev.d.wu.w $vr2, $vr6, $vr8
# CHECK-INST: vaddwev.d.wu.w $vr2, $vr6, $vr8
# CHECK-ENCODING: encoding: [0xc2,0x20,0x3f,0x70]

vaddwev.q.du.d $vr19, $vr1, $vr12
# CHECK-INST: vaddwev.q.du.d $vr19, $vr1, $vr12
# CHECK-ENCODING: encoding: [0x33,0xb0,0x3f,0x70]

vaddwod.h.b $vr31, $vr6, $vr9
# CHECK-INST: vaddwod.h.b $vr31, $vr6, $vr9
# CHECK-ENCODING: encoding: [0xdf,0x24,0x22,0x70]

vaddwod.w.h $vr17, $vr31, $vr2
# CHECK-INST: vaddwod.w.h $vr17, $vr31, $vr2
# CHECK-ENCODING: encoding: [0xf1,0x8b,0x22,0x70]

vaddwod.d.w $vr11, $vr15, $vr27
# CHECK-INST: vaddwod.d.w $vr11, $vr15, $vr27
# CHECK-ENCODING: encoding: [0xeb,0x6d,0x23,0x70]

vaddwod.q.d $vr0, $vr26, $vr17
# CHECK-INST: vaddwod.q.d $vr0, $vr26, $vr17
# CHECK-ENCODING: encoding: [0x40,0xc7,0x23,0x70]

vaddwod.h.bu $vr30, $vr15, $vr10
# CHECK-INST: vaddwod.h.bu $vr30, $vr15, $vr10
# CHECK-ENCODING: encoding: [0xfe,0x29,0x32,0x70]

vaddwod.w.hu $vr24, $vr22, $vr1
# CHECK-INST: vaddwod.w.hu $vr24, $vr22, $vr1
# CHECK-ENCODING: encoding: [0xd8,0x86,0x32,0x70]

vaddwod.d.wu $vr10, $vr25, $vr13
# CHECK-INST: vaddwod.d.wu $vr10, $vr25, $vr13
# CHECK-ENCODING: encoding: [0x2a,0x37,0x33,0x70]

vaddwod.q.du $vr16, $vr23, $vr21
# CHECK-INST: vaddwod.q.du $vr16, $vr23, $vr21
# CHECK-ENCODING: encoding: [0xf0,0xd6,0x33,0x70]

vaddwod.h.bu.b $vr30, $vr15, $vr2
# CHECK-INST: vaddwod.h.bu.b $vr30, $vr15, $vr2
# CHECK-ENCODING: encoding: [0xfe,0x09,0x40,0x70]

vaddwod.w.hu.h $vr24, $vr30, $vr13
# CHECK-INST: vaddwod.w.hu.h $vr24, $vr30, $vr13
# CHECK-ENCODING: encoding: [0xd8,0xb7,0x40,0x70]

vaddwod.d.wu.w $vr10, $vr26, $vr9
# CHECK-INST: vaddwod.d.wu.w $vr10, $vr26, $vr9
# CHECK-ENCODING: encoding: [0x4a,0x27,0x41,0x70]

vaddwod.q.du.d $vr20, $vr9, $vr16
# CHECK-INST: vaddwod.q.du.d $vr20, $vr9, $vr16
# CHECK-ENCODING: encoding: [0x34,0xc1,0x41,0x70]
