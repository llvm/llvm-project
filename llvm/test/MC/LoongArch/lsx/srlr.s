# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsrlr.b $vr27, $vr1, $vr6
# CHECK-INST: vsrlr.b $vr27, $vr1, $vr6
# CHECK-ENCODING: encoding: [0x3b,0x18,0xf0,0x70]

vsrlr.h $vr31, $vr18, $vr2
# CHECK-INST: vsrlr.h $vr31, $vr18, $vr2
# CHECK-ENCODING: encoding: [0x5f,0x8a,0xf0,0x70]

vsrlr.w $vr21, $vr29, $vr30
# CHECK-INST: vsrlr.w $vr21, $vr29, $vr30
# CHECK-ENCODING: encoding: [0xb5,0x7b,0xf1,0x70]

vsrlr.d $vr4, $vr3, $vr30
# CHECK-INST: vsrlr.d $vr4, $vr3, $vr30
# CHECK-ENCODING: encoding: [0x64,0xf8,0xf1,0x70]

vsrlri.b $vr20, $vr24, 6
# CHECK-INST: vsrlri.b $vr20, $vr24, 6
# CHECK-ENCODING: encoding: [0x14,0x3b,0xa4,0x72]

vsrlri.h $vr23, $vr22, 4
# CHECK-INST: vsrlri.h $vr23, $vr22, 4
# CHECK-ENCODING: encoding: [0xd7,0x52,0xa4,0x72]

vsrlri.w $vr19, $vr8, 1
# CHECK-INST: vsrlri.w $vr19, $vr8, 1
# CHECK-ENCODING: encoding: [0x13,0x85,0xa4,0x72]

vsrlri.d $vr18, $vr30, 51
# CHECK-INST: vsrlri.d $vr18, $vr30, 51
# CHECK-ENCODING: encoding: [0xd2,0xcf,0xa5,0x72]
