# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfcmp.caf.s $vr25, $vr5, $vr4
# CHECK-INST: vfcmp.caf.s $vr25, $vr5, $vr4
# CHECK-ENCODING: encoding: [0xb9,0x10,0x50,0x0c]

vfcmp.caf.d $vr14, $vr16, $vr23
# CHECK-INST: vfcmp.caf.d $vr14, $vr16, $vr23
# CHECK-ENCODING: encoding: [0x0e,0x5e,0x60,0x0c]

vfcmp.cun.s $vr22, $vr15, $vr4
# CHECK-INST: vfcmp.cun.s $vr22, $vr15, $vr4
# CHECK-ENCODING: encoding: [0xf6,0x11,0x54,0x0c]

vfcmp.cun.d $vr28, $vr27, $vr9
# CHECK-INST: vfcmp.cun.d $vr28, $vr27, $vr9
# CHECK-ENCODING: encoding: [0x7c,0x27,0x64,0x0c]

vfcmp.ceq.s $vr20, $vr24, $vr29
# CHECK-INST: vfcmp.ceq.s $vr20, $vr24, $vr29
# CHECK-ENCODING: encoding: [0x14,0x77,0x52,0x0c]

vfcmp.ceq.d $vr15, $vr23, $vr17
# CHECK-INST: vfcmp.ceq.d $vr15, $vr23, $vr17
# CHECK-ENCODING: encoding: [0xef,0x46,0x62,0x0c]

vfcmp.cueq.s $vr26, $vr31, $vr19
# CHECK-INST: vfcmp.cueq.s $vr26, $vr31, $vr19
# CHECK-ENCODING: encoding: [0xfa,0x4f,0x56,0x0c]

vfcmp.cueq.d $vr27, $vr10, $vr16
# CHECK-INST: vfcmp.cueq.d $vr27, $vr10, $vr16
# CHECK-ENCODING: encoding: [0x5b,0x41,0x66,0x0c]

vfcmp.clt.s $vr6, $vr27, $vr2
# CHECK-INST: vfcmp.clt.s $vr6, $vr27, $vr2
# CHECK-ENCODING: encoding: [0x66,0x0b,0x51,0x0c]

vfcmp.clt.d $vr11, $vr8, $vr6
# CHECK-INST: vfcmp.clt.d $vr11, $vr8, $vr6
# CHECK-ENCODING: encoding: [0x0b,0x19,0x61,0x0c]

vfcmp.cult.s $vr1, $vr17, $vr2
# CHECK-INST: vfcmp.cult.s $vr1, $vr17, $vr2
# CHECK-ENCODING: encoding: [0x21,0x0a,0x55,0x0c]

vfcmp.cult.d $vr11, $vr20, $vr7
# CHECK-INST: vfcmp.cult.d $vr11, $vr20, $vr7
# CHECK-ENCODING: encoding: [0x8b,0x1e,0x65,0x0c]

vfcmp.cle.s $vr10, $vr20, $vr23
# CHECK-INST: vfcmp.cle.s $vr10, $vr20, $vr23
# CHECK-ENCODING: encoding: [0x8a,0x5e,0x53,0x0c]

vfcmp.cle.d $vr1, $vr8, $vr18
# CHECK-INST: vfcmp.cle.d $vr1, $vr8, $vr18
# CHECK-ENCODING: encoding: [0x01,0x49,0x63,0x0c]

vfcmp.cule.s $vr6, $vr15, $vr11
# CHECK-INST: vfcmp.cule.s $vr6, $vr15, $vr11
# CHECK-ENCODING: encoding: [0xe6,0x2d,0x57,0x0c]

vfcmp.cule.d $vr11, $vr28, $vr30
# CHECK-INST: vfcmp.cule.d $vr11, $vr28, $vr30
# CHECK-ENCODING: encoding: [0x8b,0x7b,0x67,0x0c]

vfcmp.cne.s $vr29, $vr28, $vr11
# CHECK-INST: vfcmp.cne.s $vr29, $vr28, $vr11
# CHECK-ENCODING: encoding: [0x9d,0x2f,0x58,0x0c]

vfcmp.cne.d $vr20, $vr5, $vr7
# CHECK-INST: vfcmp.cne.d $vr20, $vr5, $vr7
# CHECK-ENCODING: encoding: [0xb4,0x1c,0x68,0x0c]

vfcmp.cor.s $vr20, $vr17, $vr12
# CHECK-INST: vfcmp.cor.s $vr20, $vr17, $vr12
# CHECK-ENCODING: encoding: [0x34,0x32,0x5a,0x0c]

vfcmp.cor.d $vr25, $vr10, $vr16
# CHECK-INST: vfcmp.cor.d $vr25, $vr10, $vr16
# CHECK-ENCODING: encoding: [0x59,0x41,0x6a,0x0c]

vfcmp.cune.s $vr26, $vr7, $vr8
# CHECK-INST: vfcmp.cune.s $vr26, $vr7, $vr8
# CHECK-ENCODING: encoding: [0xfa,0x20,0x5c,0x0c]

vfcmp.cune.d $vr13, $vr31, $vr3
# CHECK-INST: vfcmp.cune.d $vr13, $vr31, $vr3
# CHECK-ENCODING: encoding: [0xed,0x0f,0x6c,0x0c]

vfcmp.saf.s $vr26, $vr25, $vr5
# CHECK-INST: vfcmp.saf.s $vr26, $vr25, $vr5
# CHECK-ENCODING: encoding: [0x3a,0x97,0x50,0x0c]

vfcmp.saf.d $vr5, $vr29, $vr21
# CHECK-INST: vfcmp.saf.d $vr5, $vr29, $vr21
# CHECK-ENCODING: encoding: [0xa5,0xd7,0x60,0x0c]

vfcmp.sun.s $vr2, $vr2, $vr11
# CHECK-INST: vfcmp.sun.s $vr2, $vr2, $vr11
# CHECK-ENCODING: encoding: [0x42,0xac,0x54,0x0c]

vfcmp.sun.d $vr30, $vr23, $vr23
# CHECK-INST: vfcmp.sun.d $vr30, $vr23, $vr23
# CHECK-ENCODING: encoding: [0xfe,0xde,0x64,0x0c]

vfcmp.seq.s $vr4, $vr24, $vr31
# CHECK-INST: vfcmp.seq.s $vr4, $vr24, $vr31
# CHECK-ENCODING: encoding: [0x04,0xff,0x52,0x0c]

vfcmp.seq.d $vr28, $vr28, $vr5
# CHECK-INST: vfcmp.seq.d $vr28, $vr28, $vr5
# CHECK-ENCODING: encoding: [0x9c,0x97,0x62,0x0c]

vfcmp.sueq.s $vr2, $vr25, $vr29
# CHECK-INST: vfcmp.sueq.s $vr2, $vr25, $vr29
# CHECK-ENCODING: encoding: [0x22,0xf7,0x56,0x0c]

vfcmp.sueq.d $vr26, $vr16, $vr0
# CHECK-INST: vfcmp.sueq.d $vr26, $vr16, $vr0
# CHECK-ENCODING: encoding: [0x1a,0x82,0x66,0x0c]

vfcmp.slt.s $vr8, $vr22, $vr5
# CHECK-INST: vfcmp.slt.s $vr8, $vr22, $vr5
# CHECK-ENCODING: encoding: [0xc8,0x96,0x51,0x0c]

vfcmp.slt.d $vr13, $vr8, $vr22
# CHECK-INST: vfcmp.slt.d $vr13, $vr8, $vr22
# CHECK-ENCODING: encoding: [0x0d,0xd9,0x61,0x0c]

vfcmp.sult.s $vr16, $vr4, $vr21
# CHECK-INST: vfcmp.sult.s $vr16, $vr4, $vr21
# CHECK-ENCODING: encoding: [0x90,0xd4,0x55,0x0c]

vfcmp.sult.d $vr28, $vr14, $vr4
# CHECK-INST: vfcmp.sult.d $vr28, $vr14, $vr4
# CHECK-ENCODING: encoding: [0xdc,0x91,0x65,0x0c]

vfcmp.sle.s $vr13, $vr21, $vr8
# CHECK-INST: vfcmp.sle.s $vr13, $vr21, $vr8
# CHECK-ENCODING: encoding: [0xad,0xa2,0x53,0x0c]

vfcmp.sle.d $vr3, $vr18, $vr9
# CHECK-INST: vfcmp.sle.d $vr3, $vr18, $vr9
# CHECK-ENCODING: encoding: [0x43,0xa6,0x63,0x0c]

vfcmp.sule.s $vr8, $vr23, $vr19
# CHECK-INST: vfcmp.sule.s $vr8, $vr23, $vr19
# CHECK-ENCODING: encoding: [0xe8,0xce,0x57,0x0c]

vfcmp.sule.d $vr22, $vr17, $vr11
# CHECK-INST: vfcmp.sule.d $vr22, $vr17, $vr11
# CHECK-ENCODING: encoding: [0x36,0xae,0x67,0x0c]

vfcmp.sne.s $vr17, $vr25, $vr6
# CHECK-INST: vfcmp.sne.s $vr17, $vr25, $vr6
# CHECK-ENCODING: encoding: [0x31,0x9b,0x58,0x0c]

vfcmp.sne.d $vr3, $vr1, $vr28
# CHECK-INST: vfcmp.sne.d $vr3, $vr1, $vr28
# CHECK-ENCODING: encoding: [0x23,0xf0,0x68,0x0c]

vfcmp.sor.s $vr31, $vr20, $vr11
# CHECK-INST: vfcmp.sor.s $vr31, $vr20, $vr11
# CHECK-ENCODING: encoding: [0x9f,0xae,0x5a,0x0c]

vfcmp.sor.d $vr18, $vr4, $vr15
# CHECK-INST: vfcmp.sor.d $vr18, $vr4, $vr15
# CHECK-ENCODING: encoding: [0x92,0xbc,0x6a,0x0c]

vfcmp.sune.s $vr16, $vr17, $vr15
# CHECK-INST: vfcmp.sune.s $vr16, $vr17, $vr15
# CHECK-ENCODING: encoding: [0x30,0xbe,0x5c,0x0c]

vfcmp.sune.d $vr23, $vr1, $vr19
# CHECK-INST: vfcmp.sune.d $vr23, $vr1, $vr19
# CHECK-ENCODING: encoding: [0x37,0xcc,0x6c,0x0c]
