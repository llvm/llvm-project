# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vavg.b $vr13, $vr3, $vr24
# CHECK-INST: vavg.b $vr13, $vr3, $vr24
# CHECK-ENCODING: encoding: [0x6d,0x60,0x64,0x70]

vavg.h $vr3, $vr6, $vr20
# CHECK-INST: vavg.h $vr3, $vr6, $vr20
# CHECK-ENCODING: encoding: [0xc3,0xd0,0x64,0x70]

vavg.w $vr21, $vr7, $vr20
# CHECK-INST: vavg.w $vr21, $vr7, $vr20
# CHECK-ENCODING: encoding: [0xf5,0x50,0x65,0x70]

vavg.d $vr6, $vr22, $vr23
# CHECK-INST: vavg.d $vr6, $vr22, $vr23
# CHECK-ENCODING: encoding: [0xc6,0xde,0x65,0x70]

vavg.bu $vr13, $vr30, $vr16
# CHECK-INST: vavg.bu $vr13, $vr30, $vr16
# CHECK-ENCODING: encoding: [0xcd,0x43,0x66,0x70]

vavg.hu $vr0, $vr15, $vr23
# CHECK-INST: vavg.hu $vr0, $vr15, $vr23
# CHECK-ENCODING: encoding: [0xe0,0xdd,0x66,0x70]

vavg.wu $vr0, $vr17, $vr9
# CHECK-INST: vavg.wu $vr0, $vr17, $vr9
# CHECK-ENCODING: encoding: [0x20,0x26,0x67,0x70]

vavg.du $vr7, $vr22, $vr14
# CHECK-INST: vavg.du $vr7, $vr22, $vr14
# CHECK-ENCODING: encoding: [0xc7,0xba,0x67,0x70]
