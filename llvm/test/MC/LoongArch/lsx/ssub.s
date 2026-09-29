# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vssub.b $vr10, $vr10, $vr11
# CHECK-INST: vssub.b $vr10, $vr10, $vr11
# CHECK-ENCODING: encoding: [0x4a,0x2d,0x48,0x70]

vssub.h $vr2, $vr18, $vr5
# CHECK-INST: vssub.h $vr2, $vr18, $vr5
# CHECK-ENCODING: encoding: [0x42,0x96,0x48,0x70]

vssub.w $vr28, $vr10, $vr2
# CHECK-INST: vssub.w $vr28, $vr10, $vr2
# CHECK-ENCODING: encoding: [0x5c,0x09,0x49,0x70]

vssub.d $vr25, $vr3, $vr10
# CHECK-INST: vssub.d $vr25, $vr3, $vr10
# CHECK-ENCODING: encoding: [0x79,0xa8,0x49,0x70]

vssub.bu $vr31, $vr13, $vr11
# CHECK-INST: vssub.bu $vr31, $vr13, $vr11
# CHECK-ENCODING: encoding: [0xbf,0x2d,0x4c,0x70]

vssub.hu $vr15, $vr19, $vr9
# CHECK-INST: vssub.hu $vr15, $vr19, $vr9
# CHECK-ENCODING: encoding: [0x6f,0xa6,0x4c,0x70]

vssub.wu $vr15, $vr12, $vr14
# CHECK-INST: vssub.wu $vr15, $vr12, $vr14
# CHECK-ENCODING: encoding: [0x8f,0x39,0x4d,0x70]

vssub.du $vr29, $vr4, $vr11
# CHECK-INST: vssub.du $vr29, $vr4, $vr11
# CHECK-ENCODING: encoding: [0x9d,0xac,0x4d,0x70]
