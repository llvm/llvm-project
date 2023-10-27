# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vssrarni.b.h $vr3, $vr9, 2
# CHECK-INST: vssrarni.b.h $vr3, $vr9, 2
# CHECK-ENCODING: encoding: [0x23,0x49,0x68,0x73]

vssrarni.h.w $vr21, $vr17, 8
# CHECK-INST: vssrarni.h.w $vr21, $vr17, 8
# CHECK-ENCODING: encoding: [0x35,0xa2,0x68,0x73]

vssrarni.w.d $vr7, $vr6, 5
# CHECK-INST: vssrarni.w.d $vr7, $vr6, 5
# CHECK-ENCODING: encoding: [0xc7,0x14,0x69,0x73]

vssrarni.d.q $vr4, $vr22, 90
# CHECK-INST: vssrarni.d.q $vr4, $vr22, 90
# CHECK-ENCODING: encoding: [0xc4,0x6a,0x6b,0x73]

vssrarni.bu.h $vr25, $vr0, 9
# CHECK-INST: vssrarni.bu.h $vr25, $vr0, 9
# CHECK-ENCODING: encoding: [0x19,0x64,0x6c,0x73]

vssrarni.hu.w $vr5, $vr2, 24
# CHECK-INST: vssrarni.hu.w $vr5, $vr2, 24
# CHECK-ENCODING: encoding: [0x45,0xe0,0x6c,0x73]

vssrarni.wu.d $vr23, $vr29, 25
# CHECK-INST: vssrarni.wu.d $vr23, $vr29, 25
# CHECK-ENCODING: encoding: [0xb7,0x67,0x6d,0x73]

vssrarni.du.q $vr2, $vr12, 106
# CHECK-INST: vssrarni.du.q $vr2, $vr12, 106
# CHECK-ENCODING: encoding: [0x82,0xa9,0x6f,0x73]
