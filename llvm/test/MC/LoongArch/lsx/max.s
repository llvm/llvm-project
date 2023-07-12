# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmax.b $vr6, $vr21, $vr16
# CHECK-INST: vmax.b $vr6, $vr21, $vr16
# CHECK-ENCODING: encoding: [0xa6,0x42,0x70,0x70]

vmax.h $vr9, $vr28, $vr16
# CHECK-INST: vmax.h $vr9, $vr28, $vr16
# CHECK-ENCODING: encoding: [0x89,0xc3,0x70,0x70]

vmax.w $vr6, $vr0, $vr9
# CHECK-INST: vmax.w $vr6, $vr0, $vr9
# CHECK-ENCODING: encoding: [0x06,0x24,0x71,0x70]

vmax.d $vr26, $vr3, $vr0
# CHECK-INST: vmax.d $vr26, $vr3, $vr0
# CHECK-ENCODING: encoding: [0x7a,0x80,0x71,0x70]

vmaxi.b $vr2, $vr21, -8
# CHECK-INST: vmaxi.b $vr2, $vr21, -8
# CHECK-ENCODING: encoding: [0xa2,0x62,0x90,0x72]

vmaxi.h $vr2, $vr21, -2
# CHECK-INST: vmaxi.h $vr2, $vr21, -2
# CHECK-ENCODING: encoding: [0xa2,0xfa,0x90,0x72]

vmaxi.w $vr26, $vr21, -9
# CHECK-INST: vmaxi.w $vr26, $vr21, -9
# CHECK-ENCODING: encoding: [0xba,0x5e,0x91,0x72]

vmaxi.d $vr30, $vr28, -2
# CHECK-INST: vmaxi.d $vr30, $vr28, -2
# CHECK-ENCODING: encoding: [0x9e,0xfb,0x91,0x72]

vmax.bu $vr8, $vr7, $vr7
# CHECK-INST: vmax.bu $vr8, $vr7, $vr7
# CHECK-ENCODING: encoding: [0xe8,0x1c,0x74,0x70]

vmax.hu $vr21, $vr10, $vr11
# CHECK-INST: vmax.hu $vr21, $vr10, $vr11
# CHECK-ENCODING: encoding: [0x55,0xad,0x74,0x70]

vmax.wu $vr24, $vr13, $vr25
# CHECK-INST: vmax.wu $vr24, $vr13, $vr25
# CHECK-ENCODING: encoding: [0xb8,0x65,0x75,0x70]

vmax.du $vr23, $vr11, $vr14
# CHECK-INST: vmax.du $vr23, $vr11, $vr14
# CHECK-ENCODING: encoding: [0x77,0xb9,0x75,0x70]

vmaxi.bu $vr2, $vr9, 18
# CHECK-INST: vmaxi.bu $vr2, $vr9, 18
# CHECK-ENCODING: encoding: [0x22,0x49,0x94,0x72]

vmaxi.hu $vr11, $vr23, 18
# CHECK-INST: vmaxi.hu $vr11, $vr23, 18
# CHECK-ENCODING: encoding: [0xeb,0xca,0x94,0x72]

vmaxi.wu $vr15, $vr0, 29
# CHECK-INST: vmaxi.wu $vr15, $vr0, 29
# CHECK-ENCODING: encoding: [0x0f,0x74,0x95,0x72]

vmaxi.du $vr20, $vr1, 14
# CHECK-INST: vmaxi.du $vr20, $vr1, 14
# CHECK-ENCODING: encoding: [0x34,0xb8,0x95,0x72]
