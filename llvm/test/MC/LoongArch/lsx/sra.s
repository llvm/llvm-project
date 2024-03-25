# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsra.b $vr30, $vr9, $vr11
# CHECK-INST: vsra.b $vr30, $vr9, $vr11
# CHECK-ENCODING: encoding: [0x3e,0x2d,0xec,0x70]

vsra.h $vr20, $vr17, $vr26
# CHECK-INST: vsra.h $vr20, $vr17, $vr26
# CHECK-ENCODING: encoding: [0x34,0xea,0xec,0x70]

vsra.w $vr12, $vr21, $vr15
# CHECK-INST: vsra.w $vr12, $vr21, $vr15
# CHECK-ENCODING: encoding: [0xac,0x3e,0xed,0x70]

vsra.d $vr8, $vr8, $vr11
# CHECK-INST: vsra.d $vr8, $vr8, $vr11
# CHECK-ENCODING: encoding: [0x08,0xad,0xed,0x70]

vsrai.b $vr9, $vr0, 4
# CHECK-INST: vsrai.b $vr9, $vr0, 4
# CHECK-ENCODING: encoding: [0x09,0x30,0x34,0x73]

vsrai.h $vr1, $vr8, 6
# CHECK-INST: vsrai.h $vr1, $vr8, 6
# CHECK-ENCODING: encoding: [0x01,0x59,0x34,0x73]

vsrai.w $vr20, $vr30, 14
# CHECK-INST: vsrai.w $vr20, $vr30, 14
# CHECK-ENCODING: encoding: [0xd4,0xbb,0x34,0x73]

vsrai.d $vr0, $vr21, 12
# CHECK-INST: vsrai.d $vr0, $vr21, 12
# CHECK-ENCODING: encoding: [0xa0,0x32,0x35,0x73]
