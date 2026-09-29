# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsrl.b $vr20, $vr7, $vr27
# CHECK-INST: vsrl.b $vr20, $vr7, $vr27
# CHECK-ENCODING: encoding: [0xf4,0x6c,0xea,0x70]

vsrl.h $vr31, $vr5, $vr31
# CHECK-INST: vsrl.h $vr31, $vr5, $vr31
# CHECK-ENCODING: encoding: [0xbf,0xfc,0xea,0x70]

vsrl.w $vr31, $vr0, $vr6
# CHECK-INST: vsrl.w $vr31, $vr0, $vr6
# CHECK-ENCODING: encoding: [0x1f,0x18,0xeb,0x70]

vsrl.d $vr6, $vr8, $vr7
# CHECK-INST: vsrl.d $vr6, $vr8, $vr7
# CHECK-ENCODING: encoding: [0x06,0x9d,0xeb,0x70]

vsrli.b $vr17, $vr8, 6
# CHECK-INST: vsrli.b $vr17, $vr8, 6
# CHECK-ENCODING: encoding: [0x11,0x39,0x30,0x73]

vsrli.h $vr3, $vr31, 2
# CHECK-INST: vsrli.h $vr3, $vr31, 2
# CHECK-ENCODING: encoding: [0xe3,0x4b,0x30,0x73]

vsrli.w $vr17, $vr5, 0
# CHECK-INST: vsrli.w $vr17, $vr5, 0
# CHECK-ENCODING: encoding: [0xb1,0x80,0x30,0x73]

vsrli.d $vr16, $vr22, 34
# CHECK-INST: vsrli.d $vr16, $vr22, 34
# CHECK-ENCODING: encoding: [0xd0,0x8a,0x31,0x73]
