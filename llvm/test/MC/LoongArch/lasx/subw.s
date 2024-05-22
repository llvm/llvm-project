# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsubwev.h.b $xr29, $xr1, $xr28
# CHECK-INST: xvsubwev.h.b $xr29, $xr1, $xr28
# CHECK-ENCODING: encoding: [0x3d,0x70,0x20,0x74]

xvsubwev.w.h $xr24, $xr20, $xr31
# CHECK-INST: xvsubwev.w.h $xr24, $xr20, $xr31
# CHECK-ENCODING: encoding: [0x98,0xfe,0x20,0x74]

xvsubwev.d.w $xr6, $xr4, $xr11
# CHECK-INST: xvsubwev.d.w $xr6, $xr4, $xr11
# CHECK-ENCODING: encoding: [0x86,0x2c,0x21,0x74]

xvsubwev.q.d $xr27, $xr31, $xr13
# CHECK-INST: xvsubwev.q.d $xr27, $xr31, $xr13
# CHECK-ENCODING: encoding: [0xfb,0xb7,0x21,0x74]

xvsubwev.h.bu $xr1, $xr20, $xr2
# CHECK-INST: xvsubwev.h.bu $xr1, $xr20, $xr2
# CHECK-ENCODING: encoding: [0x81,0x0a,0x30,0x74]

xvsubwev.w.hu $xr19, $xr6, $xr12
# CHECK-INST: xvsubwev.w.hu $xr19, $xr6, $xr12
# CHECK-ENCODING: encoding: [0xd3,0xb0,0x30,0x74]

xvsubwev.d.wu $xr31, $xr1, $xr23
# CHECK-INST: xvsubwev.d.wu $xr31, $xr1, $xr23
# CHECK-ENCODING: encoding: [0x3f,0x5c,0x31,0x74]

xvsubwev.q.du $xr31, $xr28, $xr17
# CHECK-INST: xvsubwev.q.du $xr31, $xr28, $xr17
# CHECK-ENCODING: encoding: [0x9f,0xc7,0x31,0x74]

xvsubwod.h.b $xr3, $xr9, $xr17
# CHECK-INST: xvsubwod.h.b $xr3, $xr9, $xr17
# CHECK-ENCODING: encoding: [0x23,0x45,0x24,0x74]

xvsubwod.w.h $xr14, $xr5, $xr21
# CHECK-INST: xvsubwod.w.h $xr14, $xr5, $xr21
# CHECK-ENCODING: encoding: [0xae,0xd4,0x24,0x74]

xvsubwod.d.w $xr8, $xr14, $xr3
# CHECK-INST: xvsubwod.d.w $xr8, $xr14, $xr3
# CHECK-ENCODING: encoding: [0xc8,0x0d,0x25,0x74]

xvsubwod.q.d $xr24, $xr15, $xr18
# CHECK-INST: xvsubwod.q.d $xr24, $xr15, $xr18
# CHECK-ENCODING: encoding: [0xf8,0xc9,0x25,0x74]

xvsubwod.h.bu $xr27, $xr2, $xr1
# CHECK-INST: xvsubwod.h.bu $xr27, $xr2, $xr1
# CHECK-ENCODING: encoding: [0x5b,0x04,0x34,0x74]

xvsubwod.w.hu $xr19, $xr7, $xr22
# CHECK-INST: xvsubwod.w.hu $xr19, $xr7, $xr22
# CHECK-ENCODING: encoding: [0xf3,0xd8,0x34,0x74]

xvsubwod.d.wu $xr1, $xr24, $xr26
# CHECK-INST: xvsubwod.d.wu $xr1, $xr24, $xr26
# CHECK-ENCODING: encoding: [0x01,0x6b,0x35,0x74]

xvsubwod.q.du $xr29, $xr26, $xr7
# CHECK-INST: xvsubwod.q.du $xr29, $xr26, $xr7
# CHECK-ENCODING: encoding: [0x5d,0x9f,0x35,0x74]
