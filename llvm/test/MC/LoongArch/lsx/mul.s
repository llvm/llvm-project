# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmul.b $vr25, $vr30, $vr7
# CHECK-INST: vmul.b $vr25, $vr30, $vr7
# CHECK-ENCODING: encoding: [0xd9,0x1f,0x84,0x70]

vmul.h $vr16, $vr1, $vr26
# CHECK-INST: vmul.h $vr16, $vr1, $vr26
# CHECK-ENCODING: encoding: [0x30,0xe8,0x84,0x70]

vmul.w $vr24, $vr22, $vr29
# CHECK-INST: vmul.w $vr24, $vr22, $vr29
# CHECK-ENCODING: encoding: [0xd8,0x76,0x85,0x70]

vmul.d $vr27, $vr16, $vr25
# CHECK-INST: vmul.d $vr27, $vr16, $vr25
# CHECK-ENCODING: encoding: [0x1b,0xe6,0x85,0x70]
