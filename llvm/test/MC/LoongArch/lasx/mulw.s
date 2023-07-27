# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmulwev.h.b $xr2, $xr7, $xr16
# CHECK-INST: xvmulwev.h.b $xr2, $xr7, $xr16
# CHECK-ENCODING: encoding: [0xe2,0x40,0x90,0x74]

xvmulwev.w.h $xr12, $xr11, $xr6
# CHECK-INST: xvmulwev.w.h $xr12, $xr11, $xr6
# CHECK-ENCODING: encoding: [0x6c,0x99,0x90,0x74]

xvmulwev.d.w $xr16, $xr24, $xr15
# CHECK-INST: xvmulwev.d.w $xr16, $xr24, $xr15
# CHECK-ENCODING: encoding: [0x10,0x3f,0x91,0x74]

xvmulwev.q.d $xr17, $xr16, $xr4
# CHECK-INST: xvmulwev.q.d $xr17, $xr16, $xr4
# CHECK-ENCODING: encoding: [0x11,0x92,0x91,0x74]

xvmulwev.h.bu $xr20, $xr7, $xr29
# CHECK-INST: xvmulwev.h.bu $xr20, $xr7, $xr29
# CHECK-ENCODING: encoding: [0xf4,0x74,0x98,0x74]

xvmulwev.w.hu $xr13, $xr24, $xr17
# CHECK-INST: xvmulwev.w.hu $xr13, $xr24, $xr17
# CHECK-ENCODING: encoding: [0x0d,0xc7,0x98,0x74]

xvmulwev.d.wu $xr1, $xr24, $xr30
# CHECK-INST: xvmulwev.d.wu $xr1, $xr24, $xr30
# CHECK-ENCODING: encoding: [0x01,0x7b,0x99,0x74]

xvmulwev.q.du $xr1, $xr22, $xr27
# CHECK-INST: xvmulwev.q.du $xr1, $xr22, $xr27
# CHECK-ENCODING: encoding: [0xc1,0xee,0x99,0x74]

xvmulwev.h.bu.b $xr13, $xr28, $xr12
# CHECK-INST: xvmulwev.h.bu.b $xr13, $xr28, $xr12
# CHECK-ENCODING: encoding: [0x8d,0x33,0xa0,0x74]

xvmulwev.w.hu.h $xr27, $xr16, $xr7
# CHECK-INST: xvmulwev.w.hu.h $xr27, $xr16, $xr7
# CHECK-ENCODING: encoding: [0x1b,0x9e,0xa0,0x74]

xvmulwev.d.wu.w $xr13, $xr7, $xr17
# CHECK-INST: xvmulwev.d.wu.w $xr13, $xr7, $xr17
# CHECK-ENCODING: encoding: [0xed,0x44,0xa1,0x74]

xvmulwev.q.du.d $xr9, $xr20, $xr15
# CHECK-INST: xvmulwev.q.du.d $xr9, $xr20, $xr15
# CHECK-ENCODING: encoding: [0x89,0xbe,0xa1,0x74]

xvmulwod.h.b $xr16, $xr18, $xr2
# CHECK-INST: xvmulwod.h.b $xr16, $xr18, $xr2
# CHECK-ENCODING: encoding: [0x50,0x0a,0x92,0x74]

xvmulwod.w.h $xr30, $xr2, $xr23
# CHECK-INST: xvmulwod.w.h $xr30, $xr2, $xr23
# CHECK-ENCODING: encoding: [0x5e,0xdc,0x92,0x74]

xvmulwod.d.w $xr30, $xr27, $xr8
# CHECK-INST: xvmulwod.d.w $xr30, $xr27, $xr8
# CHECK-ENCODING: encoding: [0x7e,0x23,0x93,0x74]

xvmulwod.q.d $xr20, $xr21, $xr15
# CHECK-INST: xvmulwod.q.d $xr20, $xr21, $xr15
# CHECK-ENCODING: encoding: [0xb4,0xbe,0x93,0x74]

xvmulwod.h.bu $xr19, $xr26, $xr7
# CHECK-INST: xvmulwod.h.bu $xr19, $xr26, $xr7
# CHECK-ENCODING: encoding: [0x53,0x1f,0x9a,0x74]

xvmulwod.w.hu $xr14, $xr17, $xr6
# CHECK-INST: xvmulwod.w.hu $xr14, $xr17, $xr6
# CHECK-ENCODING: encoding: [0x2e,0x9a,0x9a,0x74]

xvmulwod.d.wu $xr24, $xr22, $xr20
# CHECK-INST: xvmulwod.d.wu $xr24, $xr22, $xr20
# CHECK-ENCODING: encoding: [0xd8,0x52,0x9b,0x74]

xvmulwod.q.du $xr28, $xr31, $xr7
# CHECK-INST: xvmulwod.q.du $xr28, $xr31, $xr7
# CHECK-ENCODING: encoding: [0xfc,0x9f,0x9b,0x74]

xvmulwod.h.bu.b $xr24, $xr15, $xr28
# CHECK-INST: xvmulwod.h.bu.b $xr24, $xr15, $xr28
# CHECK-ENCODING: encoding: [0xf8,0x71,0xa2,0x74]

xvmulwod.w.hu.h $xr24, $xr8, $xr1
# CHECK-INST: xvmulwod.w.hu.h $xr24, $xr8, $xr1
# CHECK-ENCODING: encoding: [0x18,0x85,0xa2,0x74]

xvmulwod.d.wu.w $xr10, $xr3, $xr1
# CHECK-INST: xvmulwod.d.wu.w $xr10, $xr3, $xr1
# CHECK-ENCODING: encoding: [0x6a,0x04,0xa3,0x74]

xvmulwod.q.du.d $xr15, $xr15, $xr2
# CHECK-INST: xvmulwod.q.du.d $xr15, $xr15, $xr2
# CHECK-ENCODING: encoding: [0xef,0x89,0xa3,0x74]
