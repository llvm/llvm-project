# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsrar.b $vr5, $vr31, $vr9
# CHECK-INST: vsrar.b $vr5, $vr31, $vr9
# CHECK-ENCODING: encoding: [0xe5,0x27,0xf2,0x70]

vsrar.h $vr30, $vr23, $vr30
# CHECK-INST: vsrar.h $vr30, $vr23, $vr30
# CHECK-ENCODING: encoding: [0xfe,0xfa,0xf2,0x70]

vsrar.w $vr22, $vr8, $vr1
# CHECK-INST: vsrar.w $vr22, $vr8, $vr1
# CHECK-ENCODING: encoding: [0x16,0x05,0xf3,0x70]

vsrar.d $vr17, $vr1, $vr5
# CHECK-INST: vsrar.d $vr17, $vr1, $vr5
# CHECK-ENCODING: encoding: [0x31,0x94,0xf3,0x70]

vsrari.b $vr11, $vr24, 5
# CHECK-INST: vsrari.b $vr11, $vr24, 5
# CHECK-ENCODING: encoding: [0x0b,0x37,0xa8,0x72]

vsrari.h $vr24, $vr0, 7
# CHECK-INST: vsrari.h $vr24, $vr0, 7
# CHECK-ENCODING: encoding: [0x18,0x5c,0xa8,0x72]

vsrari.w $vr16, $vr0, 0
# CHECK-INST: vsrari.w $vr16, $vr0, 0
# CHECK-ENCODING: encoding: [0x10,0x80,0xa8,0x72]

vsrari.d $vr16, $vr13, 63
# CHECK-INST: vsrari.d $vr16, $vr13, 63
# CHECK-ENCODING: encoding: [0xb0,0xfd,0xa9,0x72]
