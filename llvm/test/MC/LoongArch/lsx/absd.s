# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vabsd.b $vr14, $vr15, $vr12
# CHECK-INST: vabsd.b $vr14, $vr15, $vr12
# CHECK-ENCODING: encoding: [0xee,0x31,0x60,0x70]

vabsd.h $vr7, $vr13, $vr10
# CHECK-INST: vabsd.h $vr7, $vr13, $vr10
# CHECK-ENCODING: encoding: [0xa7,0xa9,0x60,0x70]

vabsd.w $vr5, $vr28, $vr29
# CHECK-INST: vabsd.w $vr5, $vr28, $vr29
# CHECK-ENCODING: encoding: [0x85,0x77,0x61,0x70]

vabsd.d $vr7, $vr25, $vr5
# CHECK-INST: vabsd.d $vr7, $vr25, $vr5
# CHECK-ENCODING: encoding: [0x27,0x97,0x61,0x70]

vabsd.bu $vr22, $vr16, $vr21
# CHECK-INST: vabsd.bu $vr22, $vr16, $vr21
# CHECK-ENCODING: encoding: [0x16,0x56,0x62,0x70]

vabsd.hu $vr7, $vr29, $vr8
# CHECK-INST: vabsd.hu $vr7, $vr29, $vr8
# CHECK-ENCODING: encoding: [0xa7,0xa3,0x62,0x70]

vabsd.wu $vr19, $vr31, $vr16
# CHECK-INST: vabsd.wu $vr19, $vr31, $vr16
# CHECK-ENCODING: encoding: [0xf3,0x43,0x63,0x70]

vabsd.du $vr29, $vr31, $vr17
# CHECK-INST: vabsd.du $vr29, $vr31, $vr17
# CHECK-ENCODING: encoding: [0xfd,0xc7,0x63,0x70]
