# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vftintrne.w.s $vr25, $vr28
# CHECK-INST: vftintrne.w.s $vr25, $vr28
# CHECK-ENCODING: encoding: [0x99,0x53,0x9e,0x72]

vftintrne.l.d $vr26, $vr27
# CHECK-INST: vftintrne.l.d $vr26, $vr27
# CHECK-ENCODING: encoding: [0x7a,0x57,0x9e,0x72]

vftintrz.w.s $vr24, $vr29
# CHECK-INST: vftintrz.w.s $vr24, $vr29
# CHECK-ENCODING: encoding: [0xb8,0x4b,0x9e,0x72]

vftintrz.l.d $vr17, $vr12
# CHECK-INST: vftintrz.l.d $vr17, $vr12
# CHECK-ENCODING: encoding: [0x91,0x4d,0x9e,0x72]

vftintrp.w.s $vr1, $vr6
# CHECK-INST: vftintrp.w.s $vr1, $vr6
# CHECK-ENCODING: encoding: [0xc1,0x40,0x9e,0x72]

vftintrp.l.d $vr8, $vr26
# CHECK-INST: vftintrp.l.d $vr8, $vr26
# CHECK-ENCODING: encoding: [0x48,0x47,0x9e,0x72]

vftintrm.w.s $vr4, $vr30
# CHECK-INST: vftintrm.w.s $vr4, $vr30
# CHECK-ENCODING: encoding: [0xc4,0x3b,0x9e,0x72]

vftintrm.l.d $vr18, $vr0
# CHECK-INST: vftintrm.l.d $vr18, $vr0
# CHECK-ENCODING: encoding: [0x12,0x3c,0x9e,0x72]

vftint.w.s $vr0, $vr27
# CHECK-INST: vftint.w.s $vr0, $vr27
# CHECK-ENCODING: encoding: [0x60,0x33,0x9e,0x72]

vftint.l.d $vr21, $vr22
# CHECK-INST: vftint.l.d $vr21, $vr22
# CHECK-ENCODING: encoding: [0xd5,0x36,0x9e,0x72]

vftintrz.wu.s $vr8, $vr3
# CHECK-INST: vftintrz.wu.s $vr8, $vr3
# CHECK-ENCODING: encoding: [0x68,0x70,0x9e,0x72]

vftintrz.lu.d $vr25, $vr9
# CHECK-INST: vftintrz.lu.d $vr25, $vr9
# CHECK-ENCODING: encoding: [0x39,0x75,0x9e,0x72]

vftint.wu.s $vr8, $vr8
# CHECK-INST: vftint.wu.s $vr8, $vr8
# CHECK-ENCODING: encoding: [0x08,0x59,0x9e,0x72]

vftint.lu.d $vr1, $vr17
# CHECK-INST: vftint.lu.d $vr1, $vr17
# CHECK-ENCODING: encoding: [0x21,0x5e,0x9e,0x72]

vftintrne.w.d $vr4, $vr18, $vr18
# CHECK-INST: vftintrne.w.d $vr4, $vr18, $vr18
# CHECK-ENCODING: encoding: [0x44,0xca,0x4b,0x71]

vftintrz.w.d $vr26, $vr18, $vr4
# CHECK-INST: vftintrz.w.d $vr26, $vr18, $vr4
# CHECK-ENCODING: encoding: [0x5a,0x12,0x4b,0x71]

vftintrp.w.d $vr25, $vr0, $vr23
# CHECK-INST: vftintrp.w.d $vr25, $vr0, $vr23
# CHECK-ENCODING: encoding: [0x19,0xdc,0x4a,0x71]

vftintrm.w.d $vr30, $vr25, $vr5
# CHECK-INST: vftintrm.w.d $vr30, $vr25, $vr5
# CHECK-ENCODING: encoding: [0x3e,0x17,0x4a,0x71]

vftint.w.d $vr27, $vr28, $vr6
# CHECK-INST: vftint.w.d $vr27, $vr28, $vr6
# CHECK-ENCODING: encoding: [0x9b,0x9b,0x49,0x71]

vftintrnel.l.s $vr7, $vr8
# CHECK-INST: vftintrnel.l.s $vr7, $vr8
# CHECK-ENCODING: encoding: [0x07,0xa1,0x9e,0x72]

vftintrneh.l.s $vr21, $vr26
# CHECK-INST: vftintrneh.l.s $vr21, $vr26
# CHECK-ENCODING: encoding: [0x55,0xa7,0x9e,0x72]

vftintrzl.l.s $vr21, $vr18
# CHECK-INST: vftintrzl.l.s $vr21, $vr18
# CHECK-ENCODING: encoding: [0x55,0x9a,0x9e,0x72]

vftintrzh.l.s $vr22, $vr16
# CHECK-INST: vftintrzh.l.s $vr22, $vr16
# CHECK-ENCODING: encoding: [0x16,0x9e,0x9e,0x72]

vftintrpl.l.s $vr25, $vr19
# CHECK-INST: vftintrpl.l.s $vr25, $vr19
# CHECK-ENCODING: encoding: [0x79,0x92,0x9e,0x72]

vftintrph.l.s $vr11, $vr22
# CHECK-INST: vftintrph.l.s $vr11, $vr22
# CHECK-ENCODING: encoding: [0xcb,0x96,0x9e,0x72]

vftintrml.l.s $vr6, $vr28
# CHECK-INST: vftintrml.l.s $vr6, $vr28
# CHECK-ENCODING: encoding: [0x86,0x8b,0x9e,0x72]

vftintrmh.l.s $vr17, $vr11
# CHECK-INST: vftintrmh.l.s $vr17, $vr11
# CHECK-ENCODING: encoding: [0x71,0x8d,0x9e,0x72]

vftintl.l.s $vr3, $vr28
# CHECK-INST: vftintl.l.s $vr3, $vr28
# CHECK-ENCODING: encoding: [0x83,0x83,0x9e,0x72]

vftinth.l.s $vr11, $vr30
# CHECK-INST: vftinth.l.s $vr11, $vr30
# CHECK-ENCODING: encoding: [0xcb,0x87,0x9e,0x72]
