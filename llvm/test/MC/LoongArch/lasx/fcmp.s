# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfcmp.caf.s $xr1, $xr8, $xr31
# CHECK-INST: xvfcmp.caf.s $xr1, $xr8, $xr31
# CHECK-ENCODING: encoding: [0x01,0x7d,0x90,0x0c]

xvfcmp.caf.d $xr19, $xr31, $xr20
# CHECK-INST: xvfcmp.caf.d $xr19, $xr31, $xr20
# CHECK-ENCODING: encoding: [0xf3,0x53,0xa0,0x0c]

xvfcmp.cun.s $xr8, $xr9, $xr29
# CHECK-INST: xvfcmp.cun.s $xr8, $xr9, $xr29
# CHECK-ENCODING: encoding: [0x28,0x75,0x94,0x0c]

xvfcmp.cun.d $xr19, $xr22, $xr28
# CHECK-INST: xvfcmp.cun.d $xr19, $xr22, $xr28
# CHECK-ENCODING: encoding: [0xd3,0x72,0xa4,0x0c]

xvfcmp.ceq.s $xr0, $xr1, $xr0
# CHECK-INST: xvfcmp.ceq.s $xr0, $xr1, $xr0
# CHECK-ENCODING: encoding: [0x20,0x00,0x92,0x0c]

xvfcmp.ceq.d $xr29, $xr23, $xr20
# CHECK-INST: xvfcmp.ceq.d $xr29, $xr23, $xr20
# CHECK-ENCODING: encoding: [0xfd,0x52,0xa2,0x0c]

xvfcmp.cueq.s $xr5, $xr13, $xr31
# CHECK-INST: xvfcmp.cueq.s $xr5, $xr13, $xr31
# CHECK-ENCODING: encoding: [0xa5,0x7d,0x96,0x0c]

xvfcmp.cueq.d $xr4, $xr22, $xr7
# CHECK-INST: xvfcmp.cueq.d $xr4, $xr22, $xr7
# CHECK-ENCODING: encoding: [0xc4,0x1e,0xa6,0x0c]

xvfcmp.clt.s $xr4, $xr9, $xr1
# CHECK-INST: xvfcmp.clt.s $xr4, $xr9, $xr1
# CHECK-ENCODING: encoding: [0x24,0x05,0x91,0x0c]

xvfcmp.clt.d $xr19, $xr4, $xr21
# CHECK-INST: xvfcmp.clt.d $xr19, $xr4, $xr21
# CHECK-ENCODING: encoding: [0x93,0x54,0xa1,0x0c]

xvfcmp.cult.s $xr15, $xr17, $xr3
# CHECK-INST: xvfcmp.cult.s $xr15, $xr17, $xr3
# CHECK-ENCODING: encoding: [0x2f,0x0e,0x95,0x0c]

xvfcmp.cult.d $xr20, $xr17, $xr6
# CHECK-INST: xvfcmp.cult.d $xr20, $xr17, $xr6
# CHECK-ENCODING: encoding: [0x34,0x1a,0xa5,0x0c]

xvfcmp.cle.s $xr22, $xr22, $xr15
# CHECK-INST: xvfcmp.cle.s $xr22, $xr22, $xr15
# CHECK-ENCODING: encoding: [0xd6,0x3e,0x93,0x0c]

xvfcmp.cle.d $xr21, $xr25, $xr12
# CHECK-INST: xvfcmp.cle.d $xr21, $xr25, $xr12
# CHECK-ENCODING: encoding: [0x35,0x33,0xa3,0x0c]

xvfcmp.cule.s $xr1, $xr2, $xr29
# CHECK-INST: xvfcmp.cule.s $xr1, $xr2, $xr29
# CHECK-ENCODING: encoding: [0x41,0x74,0x97,0x0c]

xvfcmp.cule.d $xr0, $xr5, $xr11
# CHECK-INST: xvfcmp.cule.d $xr0, $xr5, $xr11
# CHECK-ENCODING: encoding: [0xa0,0x2c,0xa7,0x0c]

xvfcmp.cne.s $xr7, $xr17, $xr26
# CHECK-INST: xvfcmp.cne.s $xr7, $xr17, $xr26
# CHECK-ENCODING: encoding: [0x27,0x6a,0x98,0x0c]

xvfcmp.cne.d $xr18, $xr25, $xr0
# CHECK-INST: xvfcmp.cne.d $xr18, $xr25, $xr0
# CHECK-ENCODING: encoding: [0x32,0x03,0xa8,0x0c]

xvfcmp.cor.s $xr1, $xr2, $xr14
# CHECK-INST: xvfcmp.cor.s $xr1, $xr2, $xr14
# CHECK-ENCODING: encoding: [0x41,0x38,0x9a,0x0c]

xvfcmp.cor.d $xr12, $xr19, $xr23
# CHECK-INST: xvfcmp.cor.d $xr12, $xr19, $xr23
# CHECK-ENCODING: encoding: [0x6c,0x5e,0xaa,0x0c]

xvfcmp.cune.s $xr21, $xr17, $xr4
# CHECK-INST: xvfcmp.cune.s $xr21, $xr17, $xr4
# CHECK-ENCODING: encoding: [0x35,0x12,0x9c,0x0c]

xvfcmp.cune.d $xr20, $xr30, $xr12
# CHECK-INST: xvfcmp.cune.d $xr20, $xr30, $xr12
# CHECK-ENCODING: encoding: [0xd4,0x33,0xac,0x0c]

xvfcmp.saf.s $xr23, $xr11, $xr2
# CHECK-INST: xvfcmp.saf.s $xr23, $xr11, $xr2
# CHECK-ENCODING: encoding: [0x77,0x89,0x90,0x0c]

xvfcmp.saf.d $xr7, $xr12, $xr7
# CHECK-INST: xvfcmp.saf.d $xr7, $xr12, $xr7
# CHECK-ENCODING: encoding: [0x87,0x9d,0xa0,0x0c]

xvfcmp.sun.s $xr0, $xr7, $xr30
# CHECK-INST: xvfcmp.sun.s $xr0, $xr7, $xr30
# CHECK-ENCODING: encoding: [0xe0,0xf8,0x94,0x0c]

xvfcmp.sun.d $xr4, $xr11, $xr30
# CHECK-INST: xvfcmp.sun.d $xr4, $xr11, $xr30
# CHECK-ENCODING: encoding: [0x64,0xf9,0xa4,0x0c]

xvfcmp.seq.s $xr15, $xr23, $xr27
# CHECK-INST: xvfcmp.seq.s $xr15, $xr23, $xr27
# CHECK-ENCODING: encoding: [0xef,0xee,0x92,0x0c]

xvfcmp.seq.d $xr15, $xr22, $xr3
# CHECK-INST: xvfcmp.seq.d $xr15, $xr22, $xr3
# CHECK-ENCODING: encoding: [0xcf,0x8e,0xa2,0x0c]

xvfcmp.sueq.s $xr12, $xr26, $xr9
# CHECK-INST: xvfcmp.sueq.s $xr12, $xr26, $xr9
# CHECK-ENCODING: encoding: [0x4c,0xa7,0x96,0x0c]

xvfcmp.sueq.d $xr5, $xr18, $xr17
# CHECK-INST: xvfcmp.sueq.d $xr5, $xr18, $xr17
# CHECK-ENCODING: encoding: [0x45,0xc6,0xa6,0x0c]

xvfcmp.slt.s $xr25, $xr18, $xr31
# CHECK-INST: xvfcmp.slt.s $xr25, $xr18, $xr31
# CHECK-ENCODING: encoding: [0x59,0xfe,0x91,0x0c]

xvfcmp.slt.d $xr17, $xr26, $xr24
# CHECK-INST: xvfcmp.slt.d $xr17, $xr26, $xr24
# CHECK-ENCODING: encoding: [0x51,0xe3,0xa1,0x0c]

xvfcmp.sult.s $xr8, $xr15, $xr18
# CHECK-INST: xvfcmp.sult.s $xr8, $xr15, $xr18
# CHECK-ENCODING: encoding: [0xe8,0xc9,0x95,0x0c]

xvfcmp.sult.d $xr4, $xr4, $xr5
# CHECK-INST: xvfcmp.sult.d $xr4, $xr4, $xr5
# CHECK-ENCODING: encoding: [0x84,0x94,0xa5,0x0c]

xvfcmp.sle.s $xr1, $xr5, $xr16
# CHECK-INST: xvfcmp.sle.s $xr1, $xr5, $xr16
# CHECK-ENCODING: encoding: [0xa1,0xc0,0x93,0x0c]

xvfcmp.sle.d $xr3, $xr1, $xr23
# CHECK-INST: xvfcmp.sle.d $xr3, $xr1, $xr23
# CHECK-ENCODING: encoding: [0x23,0xdc,0xa3,0x0c]

xvfcmp.sule.s $xr23, $xr11, $xr1
# CHECK-INST: xvfcmp.sule.s $xr23, $xr11, $xr1
# CHECK-ENCODING: encoding: [0x77,0x85,0x97,0x0c]

xvfcmp.sule.d $xr11, $xr10, $xr17
# CHECK-INST: xvfcmp.sule.d $xr11, $xr10, $xr17
# CHECK-ENCODING: encoding: [0x4b,0xc5,0xa7,0x0c]

xvfcmp.sne.s $xr27, $xr12, $xr30
# CHECK-INST: xvfcmp.sne.s $xr27, $xr12, $xr30
# CHECK-ENCODING: encoding: [0x9b,0xf9,0x98,0x0c]

xvfcmp.sne.d $xr20, $xr20, $xr17
# CHECK-INST: xvfcmp.sne.d $xr20, $xr20, $xr17
# CHECK-ENCODING: encoding: [0x94,0xc6,0xa8,0x0c]

xvfcmp.sor.s $xr11, $xr13, $xr2
# CHECK-INST: xvfcmp.sor.s $xr11, $xr13, $xr2
# CHECK-ENCODING: encoding: [0xab,0x89,0x9a,0x0c]

xvfcmp.sor.d $xr6, $xr28, $xr6
# CHECK-INST: xvfcmp.sor.d $xr6, $xr28, $xr6
# CHECK-ENCODING: encoding: [0x86,0x9b,0xaa,0x0c]

xvfcmp.sune.s $xr11, $xr16, $xr8
# CHECK-INST: xvfcmp.sune.s $xr11, $xr16, $xr8
# CHECK-ENCODING: encoding: [0x0b,0xa2,0x9c,0x0c]

xvfcmp.sune.d $xr30, $xr5, $xr27
# CHECK-INST: xvfcmp.sune.d $xr30, $xr5, $xr27
# CHECK-ENCODING: encoding: [0xbe,0xec,0xac,0x0c]
