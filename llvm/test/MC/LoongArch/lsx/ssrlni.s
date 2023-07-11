# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vssrlni.b.h $vr2, $vr23, 5
# CHECK-INST: vssrlni.b.h $vr2, $vr23, 5
# CHECK-ENCODING: encoding: [0xe2,0x56,0x48,0x73]

vssrlni.h.w $vr15, $vr20, 12
# CHECK-INST: vssrlni.h.w $vr15, $vr20, 12
# CHECK-ENCODING: encoding: [0x8f,0xb2,0x48,0x73]

vssrlni.w.d $vr27, $vr9, 7
# CHECK-INST: vssrlni.w.d $vr27, $vr9, 7
# CHECK-ENCODING: encoding: [0x3b,0x1d,0x49,0x73]

vssrlni.d.q $vr10, $vr2, 4
# CHECK-INST: vssrlni.d.q $vr10, $vr2, 4
# CHECK-ENCODING: encoding: [0x4a,0x10,0x4a,0x73]

vssrlni.bu.h $vr19, $vr3, 2
# CHECK-INST: vssrlni.bu.h $vr19, $vr3, 2
# CHECK-ENCODING: encoding: [0x73,0x48,0x4c,0x73]

vssrlni.hu.w $vr31, $vr19, 1
# CHECK-INST: vssrlni.hu.w $vr31, $vr19, 1
# CHECK-ENCODING: encoding: [0x7f,0x86,0x4c,0x73]

vssrlni.wu.d $vr13, $vr27, 6
# CHECK-INST: vssrlni.wu.d $vr13, $vr27, 6
# CHECK-ENCODING: encoding: [0x6d,0x1b,0x4d,0x73]

vssrlni.du.q $vr11, $vr30, 32
# CHECK-INST: vssrlni.du.q $vr11, $vr30, 32
# CHECK-ENCODING: encoding: [0xcb,0x83,0x4e,0x73]
