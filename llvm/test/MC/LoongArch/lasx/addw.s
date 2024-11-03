# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvaddwev.h.b $xr23, $xr30, $xr4
# CHECK-INST: xvaddwev.h.b $xr23, $xr30, $xr4
# CHECK-ENCODING: encoding: [0xd7,0x13,0x1e,0x74]

xvaddwev.w.h $xr20, $xr19, $xr31
# CHECK-INST: xvaddwev.w.h $xr20, $xr19, $xr31
# CHECK-ENCODING: encoding: [0x74,0xfe,0x1e,0x74]

xvaddwev.d.w $xr8, $xr9, $xr25
# CHECK-INST: xvaddwev.d.w $xr8, $xr9, $xr25
# CHECK-ENCODING: encoding: [0x28,0x65,0x1f,0x74]

xvaddwev.q.d $xr29, $xr22, $xr29
# CHECK-INST: xvaddwev.q.d $xr29, $xr22, $xr29
# CHECK-ENCODING: encoding: [0xdd,0xf6,0x1f,0x74]

xvaddwev.h.bu $xr30, $xr13, $xr26
# CHECK-INST: xvaddwev.h.bu $xr30, $xr13, $xr26
# CHECK-ENCODING: encoding: [0xbe,0x69,0x2e,0x74]

xvaddwev.w.hu $xr15, $xr31, $xr16
# CHECK-INST: xvaddwev.w.hu $xr15, $xr31, $xr16
# CHECK-ENCODING: encoding: [0xef,0xc3,0x2e,0x74]

xvaddwev.d.wu $xr16, $xr16, $xr20
# CHECK-INST: xvaddwev.d.wu $xr16, $xr16, $xr20
# CHECK-ENCODING: encoding: [0x10,0x52,0x2f,0x74]

xvaddwev.q.du $xr10, $xr18, $xr18
# CHECK-INST: xvaddwev.q.du $xr10, $xr18, $xr18
# CHECK-ENCODING: encoding: [0x4a,0xca,0x2f,0x74]

xvaddwev.h.bu.b $xr3, $xr7, $xr9
# CHECK-INST: xvaddwev.h.bu.b $xr3, $xr7, $xr9
# CHECK-ENCODING: encoding: [0xe3,0x24,0x3e,0x74]

xvaddwev.w.hu.h $xr26, $xr16, $xr27
# CHECK-INST: xvaddwev.w.hu.h $xr26, $xr16, $xr27
# CHECK-ENCODING: encoding: [0x1a,0xee,0x3e,0x74]

xvaddwev.d.wu.w $xr0, $xr13, $xr8
# CHECK-INST: xvaddwev.d.wu.w $xr0, $xr13, $xr8
# CHECK-ENCODING: encoding: [0xa0,0x21,0x3f,0x74]

xvaddwev.q.du.d $xr19, $xr10, $xr3
# CHECK-INST: xvaddwev.q.du.d $xr19, $xr10, $xr3
# CHECK-ENCODING: encoding: [0x53,0x8d,0x3f,0x74]

xvaddwod.h.b $xr14, $xr21, $xr24
# CHECK-INST: xvaddwod.h.b $xr14, $xr21, $xr24
# CHECK-ENCODING: encoding: [0xae,0x62,0x22,0x74]

xvaddwod.w.h $xr19, $xr26, $xr23
# CHECK-INST: xvaddwod.w.h $xr19, $xr26, $xr23
# CHECK-ENCODING: encoding: [0x53,0xdf,0x22,0x74]

xvaddwod.d.w $xr12, $xr9, $xr20
# CHECK-INST: xvaddwod.d.w $xr12, $xr9, $xr20
# CHECK-ENCODING: encoding: [0x2c,0x51,0x23,0x74]

xvaddwod.q.d $xr11, $xr2, $xr8
# CHECK-INST: xvaddwod.q.d $xr11, $xr2, $xr8
# CHECK-ENCODING: encoding: [0x4b,0xa0,0x23,0x74]

xvaddwod.h.bu $xr6, $xr6, $xr9
# CHECK-INST: xvaddwod.h.bu $xr6, $xr6, $xr9
# CHECK-ENCODING: encoding: [0xc6,0x24,0x32,0x74]

xvaddwod.w.hu $xr1, $xr27, $xr25
# CHECK-INST: xvaddwod.w.hu $xr1, $xr27, $xr25
# CHECK-ENCODING: encoding: [0x61,0xe7,0x32,0x74]

xvaddwod.d.wu $xr26, $xr19, $xr11
# CHECK-INST: xvaddwod.d.wu $xr26, $xr19, $xr11
# CHECK-ENCODING: encoding: [0x7a,0x2e,0x33,0x74]

xvaddwod.q.du $xr21, $xr22, $xr8
# CHECK-INST: xvaddwod.q.du $xr21, $xr22, $xr8
# CHECK-ENCODING: encoding: [0xd5,0xa2,0x33,0x74]

xvaddwod.h.bu.b $xr21, $xr26, $xr24
# CHECK-INST: xvaddwod.h.bu.b $xr21, $xr26, $xr24
# CHECK-ENCODING: encoding: [0x55,0x63,0x40,0x74]

xvaddwod.w.hu.h $xr31, $xr6, $xr16
# CHECK-INST: xvaddwod.w.hu.h $xr31, $xr6, $xr16
# CHECK-ENCODING: encoding: [0xdf,0xc0,0x40,0x74]

xvaddwod.d.wu.w $xr12, $xr28, $xr31
# CHECK-INST: xvaddwod.d.wu.w $xr12, $xr28, $xr31
# CHECK-ENCODING: encoding: [0x8c,0x7f,0x41,0x74]

xvaddwod.q.du.d $xr29, $xr4, $xr12
# CHECK-INST: xvaddwod.q.du.d $xr29, $xr4, $xr12
# CHECK-ENCODING: encoding: [0x9d,0xb0,0x41,0x74]
