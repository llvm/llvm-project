# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmod.b $vr28, $vr30, $vr25
# CHECK-INST: vmod.b $vr28, $vr30, $vr25
# CHECK-ENCODING: encoding: [0xdc,0x67,0xe2,0x70]

vmod.h $vr18, $vr31, $vr26
# CHECK-INST: vmod.h $vr18, $vr31, $vr26
# CHECK-ENCODING: encoding: [0xf2,0xeb,0xe2,0x70]

vmod.w $vr16, $vr20, $vr1
# CHECK-INST: vmod.w $vr16, $vr20, $vr1
# CHECK-ENCODING: encoding: [0x90,0x06,0xe3,0x70]

vmod.d $vr26, $vr27, $vr13
# CHECK-INST: vmod.d $vr26, $vr27, $vr13
# CHECK-ENCODING: encoding: [0x7a,0xb7,0xe3,0x70]

vmod.bu $vr19, $vr8, $vr11
# CHECK-INST: vmod.bu $vr19, $vr8, $vr11
# CHECK-ENCODING: encoding: [0x13,0x2d,0xe6,0x70]

vmod.hu $vr14, $vr21, $vr9
# CHECK-INST: vmod.hu $vr14, $vr21, $vr9
# CHECK-ENCODING: encoding: [0xae,0xa6,0xe6,0x70]

vmod.wu $vr19, $vr0, $vr5
# CHECK-INST: vmod.wu $vr19, $vr0, $vr5
# CHECK-ENCODING: encoding: [0x13,0x14,0xe7,0x70]

vmod.du $vr12, $vr18, $vr31
# CHECK-INST: vmod.du $vr12, $vr18, $vr31
# CHECK-ENCODING: encoding: [0x4c,0xfe,0xe7,0x70]
