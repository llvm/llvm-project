# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmin.b $vr24, $vr31, $vr5
# CHECK-INST: vmin.b $vr24, $vr31, $vr5
# CHECK-ENCODING: encoding: [0xf8,0x17,0x72,0x70]

vmin.h $vr8, $vr17, $vr29
# CHECK-INST: vmin.h $vr8, $vr17, $vr29
# CHECK-ENCODING: encoding: [0x28,0xf6,0x72,0x70]

vmin.w $vr6, $vr31, $vr20
# CHECK-INST: vmin.w $vr6, $vr31, $vr20
# CHECK-ENCODING: encoding: [0xe6,0x53,0x73,0x70]

vmin.d $vr5, $vr11, $vr14
# CHECK-INST: vmin.d $vr5, $vr11, $vr14
# CHECK-ENCODING: encoding: [0x65,0xb9,0x73,0x70]

vmini.b $vr8, $vr28, 0
# CHECK-INST: vmini.b $vr8, $vr28, 0
# CHECK-ENCODING: encoding: [0x88,0x03,0x92,0x72]

vmini.h $vr12, $vr12, 0
# CHECK-INST: vmini.h $vr12, $vr12, 0
# CHECK-ENCODING: encoding: [0x8c,0x81,0x92,0x72]

vmini.w $vr17, $vr1, 4
# CHECK-INST: vmini.w $vr17, $vr1, 4
# CHECK-ENCODING: encoding: [0x31,0x10,0x93,0x72]

vmini.d $vr13, $vr2, -14
# CHECK-INST: vmini.d $vr13, $vr2, -14
# CHECK-ENCODING: encoding: [0x4d,0xc8,0x93,0x72]

vmin.bu $vr30, $vr13, $vr11
# CHECK-INST: vmin.bu $vr30, $vr13, $vr11
# CHECK-ENCODING: encoding: [0xbe,0x2d,0x76,0x70]

vmin.hu $vr13, $vr10, $vr17
# CHECK-INST: vmin.hu $vr13, $vr10, $vr17
# CHECK-ENCODING: encoding: [0x4d,0xc5,0x76,0x70]

vmin.wu $vr29, $vr10, $vr27
# CHECK-INST: vmin.wu $vr29, $vr10, $vr27
# CHECK-ENCODING: encoding: [0x5d,0x6d,0x77,0x70]

vmin.du $vr8, $vr1, $vr16
# CHECK-INST: vmin.du $vr8, $vr1, $vr16
# CHECK-ENCODING: encoding: [0x28,0xc0,0x77,0x70]

vmini.bu $vr16, $vr22, 4
# CHECK-INST: vmini.bu $vr16, $vr22, 4
# CHECK-ENCODING: encoding: [0xd0,0x12,0x96,0x72]

vmini.hu $vr1, $vr24, 20
# CHECK-INST: vmini.hu $vr1, $vr24, 20
# CHECK-ENCODING: encoding: [0x01,0xd3,0x96,0x72]

vmini.wu $vr15, $vr5, 9
# CHECK-INST: vmini.wu $vr15, $vr5, 9
# CHECK-ENCODING: encoding: [0xaf,0x24,0x97,0x72]

vmini.du $vr31, $vr8, 25
# CHECK-INST: vmini.du $vr31, $vr8, 25
# CHECK-ENCODING: encoding: [0x1f,0xe5,0x97,0x72]
