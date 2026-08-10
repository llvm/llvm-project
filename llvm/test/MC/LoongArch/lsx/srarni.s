# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsrarni.b.h $vr29, $vr20, 5
# CHECK-INST: vsrarni.b.h $vr29, $vr20, 5
# CHECK-ENCODING: encoding: [0x9d,0x56,0x5c,0x73]

vsrarni.h.w $vr3, $vr29, 14
# CHECK-INST: vsrarni.h.w $vr3, $vr29, 14
# CHECK-ENCODING: encoding: [0xa3,0xbb,0x5c,0x73]

vsrarni.w.d $vr14, $vr19, 10
# CHECK-INST: vsrarni.w.d $vr14, $vr19, 10
# CHECK-ENCODING: encoding: [0x6e,0x2a,0x5d,0x73]

vsrarni.d.q $vr22, $vr27, 38
# CHECK-INST: vsrarni.d.q $vr22, $vr27, 38
# CHECK-ENCODING: encoding: [0x76,0x9b,0x5e,0x73]
