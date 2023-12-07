# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsub.b $vr7, $vr21, $vr25
# CHECK-INST: vsub.b $vr7, $vr21, $vr25
# CHECK-ENCODING: encoding: [0xa7,0x66,0x0c,0x70]

vsub.h $vr23, $vr7, $vr4
# CHECK-INST: vsub.h $vr23, $vr7, $vr4
# CHECK-ENCODING: encoding: [0xf7,0x90,0x0c,0x70]

vsub.w $vr28, $vr27, $vr25
# CHECK-INST: vsub.w $vr28, $vr27, $vr25
# CHECK-ENCODING: encoding: [0x7c,0x67,0x0d,0x70]

vsub.d $vr27, $vr11, $vr20
# CHECK-INST: vsub.d $vr27, $vr11, $vr20
# CHECK-ENCODING: encoding: [0x7b,0xd1,0x0d,0x70]

vsub.q $vr8, $vr11, $vr15
# CHECK-INST: vsub.q $vr8, $vr11, $vr15
# CHECK-ENCODING: encoding: [0x68,0xbd,0x2d,0x71]
