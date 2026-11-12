# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmaddwev.h.b $vr20, $vr27, $vr19
# CHECK-INST: vmaddwev.h.b $vr20, $vr27, $vr19
# CHECK-ENCODING: encoding: [0x74,0x4f,0xac,0x70]

vmaddwev.w.h $vr6, $vr21, $vr19
# CHECK-INST: vmaddwev.w.h $vr6, $vr21, $vr19
# CHECK-ENCODING: encoding: [0xa6,0xce,0xac,0x70]

vmaddwev.d.w $vr9, $vr20, $vr22
# CHECK-INST: vmaddwev.d.w $vr9, $vr20, $vr22
# CHECK-ENCODING: encoding: [0x89,0x5a,0xad,0x70]

vmaddwev.q.d $vr11, $vr22, $vr5
# CHECK-INST: vmaddwev.q.d $vr11, $vr22, $vr5
# CHECK-ENCODING: encoding: [0xcb,0x96,0xad,0x70]

vmaddwev.h.bu $vr7, $vr24, $vr12
# CHECK-INST: vmaddwev.h.bu $vr7, $vr24, $vr12
# CHECK-ENCODING: encoding: [0x07,0x33,0xb4,0x70]

vmaddwev.w.hu $vr14, $vr10, $vr2
# CHECK-INST: vmaddwev.w.hu $vr14, $vr10, $vr2
# CHECK-ENCODING: encoding: [0x4e,0x89,0xb4,0x70]

vmaddwev.d.wu $vr25, $vr22, $vr30
# CHECK-INST: vmaddwev.d.wu $vr25, $vr22, $vr30
# CHECK-ENCODING: encoding: [0xd9,0x7a,0xb5,0x70]

vmaddwev.q.du $vr4, $vr5, $vr10
# CHECK-INST: vmaddwev.q.du $vr4, $vr5, $vr10
# CHECK-ENCODING: encoding: [0xa4,0xa8,0xb5,0x70]

vmaddwev.h.bu.b $vr13, $vr17, $vr6
# CHECK-INST: vmaddwev.h.bu.b $vr13, $vr17, $vr6
# CHECK-ENCODING: encoding: [0x2d,0x1a,0xbc,0x70]

vmaddwev.w.hu.h $vr1, $vr29, $vr13
# CHECK-INST: vmaddwev.w.hu.h $vr1, $vr29, $vr13
# CHECK-ENCODING: encoding: [0xa1,0xb7,0xbc,0x70]

vmaddwev.d.wu.w $vr5, $vr13, $vr10
# CHECK-INST: vmaddwev.d.wu.w $vr5, $vr13, $vr10
# CHECK-ENCODING: encoding: [0xa5,0x29,0xbd,0x70]

vmaddwev.q.du.d $vr16, $vr0, $vr26
# CHECK-INST: vmaddwev.q.du.d $vr16, $vr0, $vr26
# CHECK-ENCODING: encoding: [0x10,0xe8,0xbd,0x70]

vmaddwod.h.b $vr29, $vr28, $vr11
# CHECK-INST: vmaddwod.h.b $vr29, $vr28, $vr11
# CHECK-ENCODING: encoding: [0x9d,0x2f,0xae,0x70]

vmaddwod.w.h $vr10, $vr5, $vr29
# CHECK-INST: vmaddwod.w.h $vr10, $vr5, $vr29
# CHECK-ENCODING: encoding: [0xaa,0xf4,0xae,0x70]

vmaddwod.d.w $vr16, $vr7, $vr26
# CHECK-INST: vmaddwod.d.w $vr16, $vr7, $vr26
# CHECK-ENCODING: encoding: [0xf0,0x68,0xaf,0x70]

vmaddwod.q.d $vr1, $vr4, $vr7
# CHECK-INST: vmaddwod.q.d $vr1, $vr4, $vr7
# CHECK-ENCODING: encoding: [0x81,0x9c,0xaf,0x70]

vmaddwod.h.bu $vr9, $vr28, $vr19
# CHECK-INST: vmaddwod.h.bu $vr9, $vr28, $vr19
# CHECK-ENCODING: encoding: [0x89,0x4f,0xb6,0x70]

vmaddwod.w.hu $vr4, $vr6, $vr19
# CHECK-INST: vmaddwod.w.hu $vr4, $vr6, $vr19
# CHECK-ENCODING: encoding: [0xc4,0xcc,0xb6,0x70]

vmaddwod.d.wu $vr2, $vr26, $vr26
# CHECK-INST: vmaddwod.d.wu $vr2, $vr26, $vr26
# CHECK-ENCODING: encoding: [0x42,0x6b,0xb7,0x70]

vmaddwod.q.du $vr9, $vr18, $vr31
# CHECK-INST: vmaddwod.q.du $vr9, $vr18, $vr31
# CHECK-ENCODING: encoding: [0x49,0xfe,0xb7,0x70]

vmaddwod.h.bu.b $vr22, $vr3, $vr25
# CHECK-INST: vmaddwod.h.bu.b $vr22, $vr3, $vr25
# CHECK-ENCODING: encoding: [0x76,0x64,0xbe,0x70]

vmaddwod.w.hu.h $vr17, $vr20, $vr22
# CHECK-INST: vmaddwod.w.hu.h $vr17, $vr20, $vr22
# CHECK-ENCODING: encoding: [0x91,0xda,0xbe,0x70]

vmaddwod.d.wu.w $vr21, $vr14, $vr6
# CHECK-INST: vmaddwod.d.wu.w $vr21, $vr14, $vr6
# CHECK-ENCODING: encoding: [0xd5,0x19,0xbf,0x70]

vmaddwod.q.du.d $vr8, $vr15, $vr11
# CHECK-INST: vmaddwod.q.du.d $vr8, $vr15, $vr11
# CHECK-ENCODING: encoding: [0xe8,0xad,0xbf,0x70]
