# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmulwev.h.b $vr5, $vr6, $vr0
# CHECK-INST: vmulwev.h.b $vr5, $vr6, $vr0
# CHECK-ENCODING: encoding: [0xc5,0x00,0x90,0x70]

vmulwev.w.h $vr4, $vr25, $vr2
# CHECK-INST: vmulwev.w.h $vr4, $vr25, $vr2
# CHECK-ENCODING: encoding: [0x24,0x8b,0x90,0x70]

vmulwev.d.w $vr30, $vr28, $vr27
# CHECK-INST: vmulwev.d.w $vr30, $vr28, $vr27
# CHECK-ENCODING: encoding: [0x9e,0x6f,0x91,0x70]

vmulwev.q.d $vr2, $vr1, $vr27
# CHECK-INST: vmulwev.q.d $vr2, $vr1, $vr27
# CHECK-ENCODING: encoding: [0x22,0xec,0x91,0x70]

vmulwev.h.bu $vr10, $vr9, $vr4
# CHECK-INST: vmulwev.h.bu $vr10, $vr9, $vr4
# CHECK-ENCODING: encoding: [0x2a,0x11,0x98,0x70]

vmulwev.w.hu $vr20, $vr31, $vr28
# CHECK-INST: vmulwev.w.hu $vr20, $vr31, $vr28
# CHECK-ENCODING: encoding: [0xf4,0xf3,0x98,0x70]

vmulwev.d.wu $vr4, $vr6, $vr21
# CHECK-INST: vmulwev.d.wu $vr4, $vr6, $vr21
# CHECK-ENCODING: encoding: [0xc4,0x54,0x99,0x70]

vmulwev.q.du $vr15, $vr21, $vr30
# CHECK-INST: vmulwev.q.du $vr15, $vr21, $vr30
# CHECK-ENCODING: encoding: [0xaf,0xfa,0x99,0x70]

vmulwev.h.bu.b $vr29, $vr24, $vr15
# CHECK-INST: vmulwev.h.bu.b $vr29, $vr24, $vr15
# CHECK-ENCODING: encoding: [0x1d,0x3f,0xa0,0x70]

vmulwev.w.hu.h $vr2, $vr28, $vr31
# CHECK-INST: vmulwev.w.hu.h $vr2, $vr28, $vr31
# CHECK-ENCODING: encoding: [0x82,0xff,0xa0,0x70]

vmulwev.d.wu.w $vr12, $vr23, $vr6
# CHECK-INST: vmulwev.d.wu.w $vr12, $vr23, $vr6
# CHECK-ENCODING: encoding: [0xec,0x1a,0xa1,0x70]

vmulwev.q.du.d $vr17, $vr9, $vr13
# CHECK-INST: vmulwev.q.du.d $vr17, $vr9, $vr13
# CHECK-ENCODING: encoding: [0x31,0xb5,0xa1,0x70]

vmulwod.h.b $vr17, $vr0, $vr16
# CHECK-INST: vmulwod.h.b $vr17, $vr0, $vr16
# CHECK-ENCODING: encoding: [0x11,0x40,0x92,0x70]

vmulwod.w.h $vr29, $vr5, $vr20
# CHECK-INST: vmulwod.w.h $vr29, $vr5, $vr20
# CHECK-ENCODING: encoding: [0xbd,0xd0,0x92,0x70]

vmulwod.d.w $vr7, $vr26, $vr6
# CHECK-INST: vmulwod.d.w $vr7, $vr26, $vr6
# CHECK-ENCODING: encoding: [0x47,0x1b,0x93,0x70]

vmulwod.q.d $vr13, $vr25, $vr30
# CHECK-INST: vmulwod.q.d $vr13, $vr25, $vr30
# CHECK-ENCODING: encoding: [0x2d,0xfb,0x93,0x70]

vmulwod.h.bu $vr29, $vr20, $vr10
# CHECK-INST: vmulwod.h.bu $vr29, $vr20, $vr10
# CHECK-ENCODING: encoding: [0x9d,0x2a,0x9a,0x70]

vmulwod.w.hu $vr31, $vr4, $vr25
# CHECK-INST: vmulwod.w.hu $vr31, $vr4, $vr25
# CHECK-ENCODING: encoding: [0x9f,0xe4,0x9a,0x70]

vmulwod.d.wu $vr7, $vr26, $vr16
# CHECK-INST: vmulwod.d.wu $vr7, $vr26, $vr16
# CHECK-ENCODING: encoding: [0x47,0x43,0x9b,0x70]

vmulwod.q.du $vr25, $vr10, $vr4
# CHECK-INST: vmulwod.q.du $vr25, $vr10, $vr4
# CHECK-ENCODING: encoding: [0x59,0x91,0x9b,0x70]

vmulwod.h.bu.b $vr6, $vr25, $vr11
# CHECK-INST: vmulwod.h.bu.b $vr6, $vr25, $vr11
# CHECK-ENCODING: encoding: [0x26,0x2f,0xa2,0x70]

vmulwod.w.hu.h $vr18, $vr25, $vr31
# CHECK-INST: vmulwod.w.hu.h $vr18, $vr25, $vr31
# CHECK-ENCODING: encoding: [0x32,0xff,0xa2,0x70]

vmulwod.d.wu.w $vr10, $vr28, $vr26
# CHECK-INST: vmulwod.d.wu.w $vr10, $vr28, $vr26
# CHECK-ENCODING: encoding: [0x8a,0x6b,0xa3,0x70]

vmulwod.q.du.d $vr30, $vr23, $vr17
# CHECK-INST: vmulwod.q.du.d $vr30, $vr23, $vr17
# CHECK-ENCODING: encoding: [0xfe,0xc6,0xa3,0x70]
