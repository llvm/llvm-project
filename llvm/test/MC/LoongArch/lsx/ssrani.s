# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vssrani.b.h $vr3, $vr12, 10
# CHECK-INST: vssrani.b.h $vr3, $vr12, 10
# CHECK-ENCODING: encoding: [0x83,0x69,0x60,0x73]

vssrani.h.w $vr3, $vr25, 0
# CHECK-INST: vssrani.h.w $vr3, $vr25, 0
# CHECK-ENCODING: encoding: [0x23,0x83,0x60,0x73]

vssrani.w.d $vr12, $vr19, 43
# CHECK-INST: vssrani.w.d $vr12, $vr19, 43
# CHECK-ENCODING: encoding: [0x6c,0xae,0x61,0x73]

vssrani.d.q $vr25, $vr8, 13
# CHECK-INST: vssrani.d.q $vr25, $vr8, 13
# CHECK-ENCODING: encoding: [0x19,0x35,0x62,0x73]

vssrani.bu.h $vr26, $vr16, 12
# CHECK-INST: vssrani.bu.h $vr26, $vr16, 12
# CHECK-ENCODING: encoding: [0x1a,0x72,0x64,0x73]

vssrani.hu.w $vr31, $vr6, 28
# CHECK-INST: vssrani.hu.w $vr31, $vr6, 28
# CHECK-ENCODING: encoding: [0xdf,0xf0,0x64,0x73]

vssrani.wu.d $vr29, $vr25, 2
# CHECK-INST: vssrani.wu.d $vr29, $vr25, 2
# CHECK-ENCODING: encoding: [0x3d,0x0b,0x65,0x73]

vssrani.du.q $vr22, $vr27, 71
# CHECK-INST: vssrani.du.q $vr22, $vr27, 71
# CHECK-ENCODING: encoding: [0x76,0x1f,0x67,0x73]
