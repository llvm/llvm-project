# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvftintrne.w.s $xr20, $xr13
# CHECK-INST: xvftintrne.w.s $xr20, $xr13
# CHECK-ENCODING: encoding: [0xb4,0x51,0x9e,0x76]

xvftintrne.l.d $xr30, $xr14
# CHECK-INST: xvftintrne.l.d $xr30, $xr14
# CHECK-ENCODING: encoding: [0xde,0x55,0x9e,0x76]

xvftintrz.w.s $xr14, $xr5
# CHECK-INST: xvftintrz.w.s $xr14, $xr5
# CHECK-ENCODING: encoding: [0xae,0x48,0x9e,0x76]

xvftintrz.l.d $xr1, $xr26
# CHECK-INST: xvftintrz.l.d $xr1, $xr26
# CHECK-ENCODING: encoding: [0x41,0x4f,0x9e,0x76]

xvftintrp.w.s $xr18, $xr1
# CHECK-INST: xvftintrp.w.s $xr18, $xr1
# CHECK-ENCODING: encoding: [0x32,0x40,0x9e,0x76]

xvftintrp.l.d $xr10, $xr24
# CHECK-INST: xvftintrp.l.d $xr10, $xr24
# CHECK-ENCODING: encoding: [0x0a,0x47,0x9e,0x76]

xvftintrm.w.s $xr8, $xr23
# CHECK-INST: xvftintrm.w.s $xr8, $xr23
# CHECK-ENCODING: encoding: [0xe8,0x3a,0x9e,0x76]

xvftintrm.l.d $xr12, $xr17
# CHECK-INST: xvftintrm.l.d $xr12, $xr17
# CHECK-ENCODING: encoding: [0x2c,0x3e,0x9e,0x76]

xvftint.w.s $xr11, $xr25
# CHECK-INST: xvftint.w.s $xr11, $xr25
# CHECK-ENCODING: encoding: [0x2b,0x33,0x9e,0x76]

xvftint.l.d $xr7, $xr22
# CHECK-INST: xvftint.l.d $xr7, $xr22
# CHECK-ENCODING: encoding: [0xc7,0x36,0x9e,0x76]

xvftintrz.wu.s $xr13, $xr19
# CHECK-INST: xvftintrz.wu.s $xr13, $xr19
# CHECK-ENCODING: encoding: [0x6d,0x72,0x9e,0x76]

xvftintrz.lu.d $xr24, $xr3
# CHECK-INST: xvftintrz.lu.d $xr24, $xr3
# CHECK-ENCODING: encoding: [0x78,0x74,0x9e,0x76]

xvftint.wu.s $xr14, $xr6
# CHECK-INST: xvftint.wu.s $xr14, $xr6
# CHECK-ENCODING: encoding: [0xce,0x58,0x9e,0x76]

xvftint.lu.d $xr2, $xr2
# CHECK-INST: xvftint.lu.d $xr2, $xr2
# CHECK-ENCODING: encoding: [0x42,0x5c,0x9e,0x76]

xvftintrne.w.d $xr13, $xr20, $xr5
# CHECK-INST: xvftintrne.w.d $xr13, $xr20, $xr5
# CHECK-ENCODING: encoding: [0x8d,0x96,0x4b,0x75]

xvftintrz.w.d $xr13, $xr8, $xr27
# CHECK-INST: xvftintrz.w.d $xr13, $xr8, $xr27
# CHECK-ENCODING: encoding: [0x0d,0x6d,0x4b,0x75]

xvftintrp.w.d $xr14, $xr26, $xr31
# CHECK-INST: xvftintrp.w.d $xr14, $xr26, $xr31
# CHECK-ENCODING: encoding: [0x4e,0xff,0x4a,0x75]

xvftintrm.w.d $xr29, $xr23, $xr7
# CHECK-INST: xvftintrm.w.d $xr29, $xr23, $xr7
# CHECK-ENCODING: encoding: [0xfd,0x1e,0x4a,0x75]

xvftint.w.d $xr7, $xr22, $xr29
# CHECK-INST: xvftint.w.d $xr7, $xr22, $xr29
# CHECK-ENCODING: encoding: [0xc7,0xf6,0x49,0x75]

xvftintrnel.l.s $xr31, $xr28
# CHECK-INST: xvftintrnel.l.s $xr31, $xr28
# CHECK-ENCODING: encoding: [0x9f,0xa3,0x9e,0x76]

xvftintrneh.l.s $xr16, $xr29
# CHECK-INST: xvftintrneh.l.s $xr16, $xr29
# CHECK-ENCODING: encoding: [0xb0,0xa7,0x9e,0x76]

xvftintrzl.l.s $xr27, $xr29
# CHECK-INST: xvftintrzl.l.s $xr27, $xr29
# CHECK-ENCODING: encoding: [0xbb,0x9b,0x9e,0x76]

xvftintrzh.l.s $xr14, $xr10
# CHECK-INST: xvftintrzh.l.s $xr14, $xr10
# CHECK-ENCODING: encoding: [0x4e,0x9d,0x9e,0x76]

xvftintrpl.l.s $xr14, $xr0
# CHECK-INST: xvftintrpl.l.s $xr14, $xr0
# CHECK-ENCODING: encoding: [0x0e,0x90,0x9e,0x76]

xvftintrph.l.s $xr23, $xr0
# CHECK-INST: xvftintrph.l.s $xr23, $xr0
# CHECK-ENCODING: encoding: [0x17,0x94,0x9e,0x76]

xvftintrml.l.s $xr22, $xr15
# CHECK-INST: xvftintrml.l.s $xr22, $xr15
# CHECK-ENCODING: encoding: [0xf6,0x89,0x9e,0x76]

xvftintrmh.l.s $xr10, $xr19
# CHECK-INST: xvftintrmh.l.s $xr10, $xr19
# CHECK-ENCODING: encoding: [0x6a,0x8e,0x9e,0x76]

xvftintl.l.s $xr31, $xr11
# CHECK-INST: xvftintl.l.s $xr31, $xr11
# CHECK-ENCODING: encoding: [0x7f,0x81,0x9e,0x76]

xvftinth.l.s $xr15, $xr5
# CHECK-INST: xvftinth.l.s $xr15, $xr5
# CHECK-ENCODING: encoding: [0xaf,0x84,0x9e,0x76]
