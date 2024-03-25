# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vavgr.b $vr22, $vr3, $vr9
# CHECK-INST: vavgr.b $vr22, $vr3, $vr9
# CHECK-ENCODING: encoding: [0x76,0x24,0x68,0x70]

vavgr.h $vr12, $vr2, $vr6
# CHECK-INST: vavgr.h $vr12, $vr2, $vr6
# CHECK-ENCODING: encoding: [0x4c,0x98,0x68,0x70]

vavgr.w $vr16, $vr30, $vr13
# CHECK-INST: vavgr.w $vr16, $vr30, $vr13
# CHECK-ENCODING: encoding: [0xd0,0x37,0x69,0x70]

vavgr.d $vr5, $vr18, $vr7
# CHECK-INST: vavgr.d $vr5, $vr18, $vr7
# CHECK-ENCODING: encoding: [0x45,0x9e,0x69,0x70]

vavgr.bu $vr22, $vr5, $vr29
# CHECK-INST: vavgr.bu $vr22, $vr5, $vr29
# CHECK-ENCODING: encoding: [0xb6,0x74,0x6a,0x70]

vavgr.hu $vr22, $vr23, $vr8
# CHECK-INST: vavgr.hu $vr22, $vr23, $vr8
# CHECK-ENCODING: encoding: [0xf6,0xa2,0x6a,0x70]

vavgr.wu $vr10, $vr20, $vr21
# CHECK-INST: vavgr.wu $vr10, $vr20, $vr21
# CHECK-ENCODING: encoding: [0x8a,0x56,0x6b,0x70]

vavgr.du $vr10, $vr28, $vr13
# CHECK-INST: vavgr.du $vr10, $vr28, $vr13
# CHECK-ENCODING: encoding: [0x8a,0xb7,0x6b,0x70]
