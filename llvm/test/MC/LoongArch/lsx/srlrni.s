# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsrlrni.b.h $vr15, $vr5, 3
# CHECK-INST: vsrlrni.b.h $vr15, $vr5, 3
# CHECK-ENCODING: encoding: [0xaf,0x4c,0x44,0x73]

vsrlrni.h.w $vr28, $vr27, 1
# CHECK-INST: vsrlrni.h.w $vr28, $vr27, 1
# CHECK-ENCODING: encoding: [0x7c,0x87,0x44,0x73]

vsrlrni.w.d $vr3, $vr25, 56
# CHECK-INST: vsrlrni.w.d $vr3, $vr25, 56
# CHECK-ENCODING: encoding: [0x23,0xe3,0x45,0x73]

vsrlrni.d.q $vr4, $vr16, 13
# CHECK-INST: vsrlrni.d.q $vr4, $vr16, 13
# CHECK-ENCODING: encoding: [0x04,0x36,0x46,0x73]
