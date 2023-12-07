# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsubwev.h.b $vr21, $vr25, $vr20
# CHECK-INST: vsubwev.h.b $vr21, $vr25, $vr20
# CHECK-ENCODING: encoding: [0x35,0x53,0x20,0x70]

vsubwev.w.h $vr11, $vr8, $vr10
# CHECK-INST: vsubwev.w.h $vr11, $vr8, $vr10
# CHECK-ENCODING: encoding: [0x0b,0xa9,0x20,0x70]

vsubwev.d.w $vr30, $vr6, $vr24
# CHECK-INST: vsubwev.d.w $vr30, $vr6, $vr24
# CHECK-ENCODING: encoding: [0xde,0x60,0x21,0x70]

vsubwev.q.d $vr4, $vr14, $vr23
# CHECK-INST: vsubwev.q.d $vr4, $vr14, $vr23
# CHECK-ENCODING: encoding: [0xc4,0xdd,0x21,0x70]

vsubwev.h.bu $vr25, $vr20, $vr2
# CHECK-INST: vsubwev.h.bu $vr25, $vr20, $vr2
# CHECK-ENCODING: encoding: [0x99,0x0a,0x30,0x70]

vsubwev.w.hu $vr1, $vr9, $vr28
# CHECK-INST: vsubwev.w.hu $vr1, $vr9, $vr28
# CHECK-ENCODING: encoding: [0x21,0xf1,0x30,0x70]

vsubwev.d.wu $vr23, $vr13, $vr2
# CHECK-INST: vsubwev.d.wu $vr23, $vr13, $vr2
# CHECK-ENCODING: encoding: [0xb7,0x09,0x31,0x70]

vsubwev.q.du $vr9, $vr28, $vr12
# CHECK-INST: vsubwev.q.du $vr9, $vr28, $vr12
# CHECK-ENCODING: encoding: [0x89,0xb3,0x31,0x70]

vsubwod.h.b $vr9, $vr12, $vr26
# CHECK-INST: vsubwod.h.b $vr9, $vr12, $vr26
# CHECK-ENCODING: encoding: [0x89,0x69,0x24,0x70]

vsubwod.w.h $vr31, $vr2, $vr10
# CHECK-INST: vsubwod.w.h $vr31, $vr2, $vr10
# CHECK-ENCODING: encoding: [0x5f,0xa8,0x24,0x70]

vsubwod.d.w $vr6, $vr16, $vr15
# CHECK-INST: vsubwod.d.w $vr6, $vr16, $vr15
# CHECK-ENCODING: encoding: [0x06,0x3e,0x25,0x70]

vsubwod.q.d $vr22, $vr0, $vr18
# CHECK-INST: vsubwod.q.d $vr22, $vr0, $vr18
# CHECK-ENCODING: encoding: [0x16,0xc8,0x25,0x70]

vsubwod.h.bu $vr3, $vr17, $vr11
# CHECK-INST: vsubwod.h.bu $vr3, $vr17, $vr11
# CHECK-ENCODING: encoding: [0x23,0x2e,0x34,0x70]

vsubwod.w.hu $vr9, $vr16, $vr26
# CHECK-INST: vsubwod.w.hu $vr9, $vr16, $vr26
# CHECK-ENCODING: encoding: [0x09,0xea,0x34,0x70]

vsubwod.d.wu $vr23, $vr9, $vr8
# CHECK-INST: vsubwod.d.wu $vr23, $vr9, $vr8
# CHECK-ENCODING: encoding: [0x37,0x21,0x35,0x70]

vsubwod.q.du $vr8, $vr15, $vr7
# CHECK-INST: vsubwod.q.du $vr8, $vr15, $vr7
# CHECK-ENCODING: encoding: [0xe8,0x9d,0x35,0x70]
