# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmaddwev.h.b $xr25, $xr15, $xr9
# CHECK-INST: xvmaddwev.h.b $xr25, $xr15, $xr9
# CHECK-ENCODING: encoding: [0xf9,0x25,0xac,0x74]

xvmaddwev.w.h $xr26, $xr1, $xr0
# CHECK-INST: xvmaddwev.w.h $xr26, $xr1, $xr0
# CHECK-ENCODING: encoding: [0x3a,0x80,0xac,0x74]

xvmaddwev.d.w $xr23, $xr24, $xr24
# CHECK-INST: xvmaddwev.d.w $xr23, $xr24, $xr24
# CHECK-ENCODING: encoding: [0x17,0x63,0xad,0x74]

xvmaddwev.q.d $xr7, $xr9, $xr22
# CHECK-INST: xvmaddwev.q.d $xr7, $xr9, $xr22
# CHECK-ENCODING: encoding: [0x27,0xd9,0xad,0x74]

xvmaddwev.h.bu $xr23, $xr13, $xr26
# CHECK-INST: xvmaddwev.h.bu $xr23, $xr13, $xr26
# CHECK-ENCODING: encoding: [0xb7,0x69,0xb4,0x74]

xvmaddwev.w.hu $xr13, $xr3, $xr3
# CHECK-INST: xvmaddwev.w.hu $xr13, $xr3, $xr3
# CHECK-ENCODING: encoding: [0x6d,0x8c,0xb4,0x74]

xvmaddwev.d.wu $xr29, $xr27, $xr28
# CHECK-INST: xvmaddwev.d.wu $xr29, $xr27, $xr28
# CHECK-ENCODING: encoding: [0x7d,0x73,0xb5,0x74]

xvmaddwev.q.du $xr29, $xr10, $xr10
# CHECK-INST: xvmaddwev.q.du $xr29, $xr10, $xr10
# CHECK-ENCODING: encoding: [0x5d,0xa9,0xb5,0x74]

xvmaddwev.h.bu.b $xr30, $xr26, $xr31
# CHECK-INST: xvmaddwev.h.bu.b $xr30, $xr26, $xr31
# CHECK-ENCODING: encoding: [0x5e,0x7f,0xbc,0x74]

xvmaddwev.w.hu.h $xr6, $xr17, $xr31
# CHECK-INST: xvmaddwev.w.hu.h $xr6, $xr17, $xr31
# CHECK-ENCODING: encoding: [0x26,0xfe,0xbc,0x74]

xvmaddwev.d.wu.w $xr10, $xr28, $xr2
# CHECK-INST: xvmaddwev.d.wu.w $xr10, $xr28, $xr2
# CHECK-ENCODING: encoding: [0x8a,0x0b,0xbd,0x74]

xvmaddwev.q.du.d $xr16, $xr20, $xr24
# CHECK-INST: xvmaddwev.q.du.d $xr16, $xr20, $xr24
# CHECK-ENCODING: encoding: [0x90,0xe2,0xbd,0x74]

xvmaddwod.h.b $xr16, $xr8, $xr18
# CHECK-INST: xvmaddwod.h.b $xr16, $xr8, $xr18
# CHECK-ENCODING: encoding: [0x10,0x49,0xae,0x74]

xvmaddwod.w.h $xr11, $xr24, $xr14
# CHECK-INST: xvmaddwod.w.h $xr11, $xr24, $xr14
# CHECK-ENCODING: encoding: [0x0b,0xbb,0xae,0x74]

xvmaddwod.d.w $xr0, $xr20, $xr13
# CHECK-INST: xvmaddwod.d.w $xr0, $xr20, $xr13
# CHECK-ENCODING: encoding: [0x80,0x36,0xaf,0x74]

xvmaddwod.q.d $xr15, $xr23, $xr18
# CHECK-INST: xvmaddwod.q.d $xr15, $xr23, $xr18
# CHECK-ENCODING: encoding: [0xef,0xca,0xaf,0x74]

xvmaddwod.h.bu $xr31, $xr23, $xr7
# CHECK-INST: xvmaddwod.h.bu $xr31, $xr23, $xr7
# CHECK-ENCODING: encoding: [0xff,0x1e,0xb6,0x74]

xvmaddwod.w.hu $xr29, $xr16, $xr8
# CHECK-INST: xvmaddwod.w.hu $xr29, $xr16, $xr8
# CHECK-ENCODING: encoding: [0x1d,0xa2,0xb6,0x74]

xvmaddwod.d.wu $xr23, $xr16, $xr11
# CHECK-INST: xvmaddwod.d.wu $xr23, $xr16, $xr11
# CHECK-ENCODING: encoding: [0x17,0x2e,0xb7,0x74]

xvmaddwod.q.du $xr9, $xr10, $xr19
# CHECK-INST: xvmaddwod.q.du $xr9, $xr10, $xr19
# CHECK-ENCODING: encoding: [0x49,0xcd,0xb7,0x74]

xvmaddwod.h.bu.b $xr27, $xr2, $xr11
# CHECK-INST: xvmaddwod.h.bu.b $xr27, $xr2, $xr11
# CHECK-ENCODING: encoding: [0x5b,0x2c,0xbe,0x74]

xvmaddwod.w.hu.h $xr12, $xr24, $xr19
# CHECK-INST: xvmaddwod.w.hu.h $xr12, $xr24, $xr19
# CHECK-ENCODING: encoding: [0x0c,0xcf,0xbe,0x74]

xvmaddwod.d.wu.w $xr11, $xr0, $xr14
# CHECK-INST: xvmaddwod.d.wu.w $xr11, $xr0, $xr14
# CHECK-ENCODING: encoding: [0x0b,0x38,0xbf,0x74]

xvmaddwod.q.du.d $xr29, $xr19, $xr31
# CHECK-INST: xvmaddwod.q.du.d $xr29, $xr19, $xr31
# CHECK-ENCODING: encoding: [0x7d,0xfe,0xbf,0x74]
