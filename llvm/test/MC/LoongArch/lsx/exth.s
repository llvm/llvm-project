# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vexth.h.b $vr9, $vr6
# CHECK-INST: vexth.h.b $vr9, $vr6
# CHECK-ENCODING: encoding: [0xc9,0xe0,0x9e,0x72]

vexth.w.h $vr14, $vr19
# CHECK-INST: vexth.w.h $vr14, $vr19
# CHECK-ENCODING: encoding: [0x6e,0xe6,0x9e,0x72]

vexth.d.w $vr1, $vr20
# CHECK-INST: vexth.d.w $vr1, $vr20
# CHECK-ENCODING: encoding: [0x81,0xea,0x9e,0x72]

vexth.q.d $vr20, $vr10
# CHECK-INST: vexth.q.d $vr20, $vr10
# CHECK-ENCODING: encoding: [0x54,0xed,0x9e,0x72]

vexth.hu.bu $vr5, $vr1
# CHECK-INST: vexth.hu.bu $vr5, $vr1
# CHECK-ENCODING: encoding: [0x25,0xf0,0x9e,0x72]

vexth.wu.hu $vr17, $vr26
# CHECK-INST: vexth.wu.hu $vr17, $vr26
# CHECK-ENCODING: encoding: [0x51,0xf7,0x9e,0x72]

vexth.du.wu $vr2, $vr7
# CHECK-INST: vexth.du.wu $vr2, $vr7
# CHECK-ENCODING: encoding: [0xe2,0xf8,0x9e,0x72]

vexth.qu.du $vr19, $vr11
# CHECK-INST: vexth.qu.du $vr19, $vr11
# CHECK-ENCODING: encoding: [0x73,0xfd,0x9e,0x72]
