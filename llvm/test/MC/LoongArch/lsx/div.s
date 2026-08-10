# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vdiv.b $vr26, $vr17, $vr24
# CHECK-INST: vdiv.b $vr26, $vr17, $vr24
# CHECK-ENCODING: encoding: [0x3a,0x62,0xe0,0x70]

vdiv.h $vr26, $vr23, $vr21
# CHECK-INST: vdiv.h $vr26, $vr23, $vr21
# CHECK-ENCODING: encoding: [0xfa,0xd6,0xe0,0x70]

vdiv.w $vr1, $vr13, $vr10
# CHECK-INST: vdiv.w $vr1, $vr13, $vr10
# CHECK-ENCODING: encoding: [0xa1,0x29,0xe1,0x70]

vdiv.d $vr4, $vr25, $vr21
# CHECK-INST: vdiv.d $vr4, $vr25, $vr21
# CHECK-ENCODING: encoding: [0x24,0xd7,0xe1,0x70]

vdiv.bu $vr13, $vr13, $vr6
# CHECK-INST: vdiv.bu $vr13, $vr13, $vr6
# CHECK-ENCODING: encoding: [0xad,0x19,0xe4,0x70]

vdiv.hu $vr1, $vr30, $vr5
# CHECK-INST: vdiv.hu $vr1, $vr30, $vr5
# CHECK-ENCODING: encoding: [0xc1,0x97,0xe4,0x70]

vdiv.wu $vr27, $vr31, $vr20
# CHECK-INST: vdiv.wu $vr27, $vr31, $vr20
# CHECK-ENCODING: encoding: [0xfb,0x53,0xe5,0x70]

vdiv.du $vr30, $vr0, $vr5
# CHECK-INST: vdiv.du $vr30, $vr0, $vr5
# CHECK-ENCODING: encoding: [0x1e,0x94,0xe5,0x70]
