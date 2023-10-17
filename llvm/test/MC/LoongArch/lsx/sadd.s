# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsadd.b $vr29, $vr30, $vr11
# CHECK-INST: vsadd.b $vr29, $vr30, $vr11
# CHECK-ENCODING: encoding: [0xdd,0x2f,0x46,0x70]

vsadd.h $vr1, $vr2, $vr29
# CHECK-INST: vsadd.h $vr1, $vr2, $vr29
# CHECK-ENCODING: encoding: [0x41,0xf4,0x46,0x70]

vsadd.w $vr19, $vr28, $vr28
# CHECK-INST: vsadd.w $vr19, $vr28, $vr28
# CHECK-ENCODING: encoding: [0x93,0x73,0x47,0x70]

vsadd.d $vr19, $vr30, $vr20
# CHECK-INST: vsadd.d $vr19, $vr30, $vr20
# CHECK-ENCODING: encoding: [0xd3,0xd3,0x47,0x70]

vsadd.bu $vr22, $vr22, $vr16
# CHECK-INST: vsadd.bu $vr22, $vr22, $vr16
# CHECK-ENCODING: encoding: [0xd6,0x42,0x4a,0x70]

vsadd.hu $vr0, $vr16, $vr8
# CHECK-INST: vsadd.hu $vr0, $vr16, $vr8
# CHECK-ENCODING: encoding: [0x00,0xa2,0x4a,0x70]

vsadd.wu $vr9, $vr23, $vr24
# CHECK-INST: vsadd.wu $vr9, $vr23, $vr24
# CHECK-ENCODING: encoding: [0xe9,0x62,0x4b,0x70]

vsadd.du $vr28, $vr11, $vr30
# CHECK-INST: vsadd.du $vr28, $vr11, $vr30
# CHECK-ENCODING: encoding: [0x7c,0xf9,0x4b,0x70]
