# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfrintrne.s $vr31, $vr2
# CHECK-INST: vfrintrne.s $vr31, $vr2
# CHECK-ENCODING: encoding: [0x5f,0x74,0x9d,0x72]

vfrintrne.d $vr1, $vr30
# CHECK-INST: vfrintrne.d $vr1, $vr30
# CHECK-ENCODING: encoding: [0xc1,0x7b,0x9d,0x72]

vfrintrz.s $vr16, $vr17
# CHECK-INST: vfrintrz.s $vr16, $vr17
# CHECK-ENCODING: encoding: [0x30,0x66,0x9d,0x72]

vfrintrz.d $vr1, $vr31
# CHECK-INST: vfrintrz.d $vr1, $vr31
# CHECK-ENCODING: encoding: [0xe1,0x6b,0x9d,0x72]

vfrintrp.s $vr11, $vr2
# CHECK-INST: vfrintrp.s $vr11, $vr2
# CHECK-ENCODING: encoding: [0x4b,0x54,0x9d,0x72]

vfrintrp.d $vr30, $vr16
# CHECK-INST: vfrintrp.d $vr30, $vr16
# CHECK-ENCODING: encoding: [0x1e,0x5a,0x9d,0x72]

vfrintrm.s $vr25, $vr23
# CHECK-INST: vfrintrm.s $vr25, $vr23
# CHECK-ENCODING: encoding: [0xf9,0x46,0x9d,0x72]

vfrintrm.d $vr19, $vr11
# CHECK-INST: vfrintrm.d $vr19, $vr11
# CHECK-ENCODING: encoding: [0x73,0x49,0x9d,0x72]

vfrint.s $vr22, $vr6
# CHECK-INST: vfrint.s $vr22, $vr6
# CHECK-ENCODING: encoding: [0xd6,0x34,0x9d,0x72]

vfrint.d $vr26, $vr9
# CHECK-INST: vfrint.d $vr26, $vr9
# CHECK-ENCODING: encoding: [0x3a,0x39,0x9d,0x72]
