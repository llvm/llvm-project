# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vssrlrni.b.h $vr18, $vr21, 6
# CHECK-INST: vssrlrni.b.h $vr18, $vr21, 6
# CHECK-ENCODING: encoding: [0xb2,0x5a,0x50,0x73]

vssrlrni.h.w $vr7, $vr12, 9
# CHECK-INST: vssrlrni.h.w $vr7, $vr12, 9
# CHECK-ENCODING: encoding: [0x87,0xa5,0x50,0x73]

vssrlrni.w.d $vr10, $vr14, 63
# CHECK-INST: vssrlrni.w.d $vr10, $vr14, 63
# CHECK-ENCODING: encoding: [0xca,0xfd,0x51,0x73]

vssrlrni.d.q $vr12, $vr26, 68
# CHECK-INST: vssrlrni.d.q $vr12, $vr26, 68
# CHECK-ENCODING: encoding: [0x4c,0x13,0x53,0x73]

vssrlrni.bu.h $vr22, $vr24, 1
# CHECK-INST: vssrlrni.bu.h $vr22, $vr24, 1
# CHECK-ENCODING: encoding: [0x16,0x47,0x54,0x73]

vssrlrni.hu.w $vr27, $vr17, 7
# CHECK-INST: vssrlrni.hu.w $vr27, $vr17, 7
# CHECK-ENCODING: encoding: [0x3b,0x9e,0x54,0x73]

vssrlrni.wu.d $vr3, $vr15, 56
# CHECK-INST: vssrlrni.wu.d $vr3, $vr15, 56
# CHECK-ENCODING: encoding: [0xe3,0xe1,0x55,0x73]

vssrlrni.du.q $vr12, $vr10, 4
# CHECK-INST: vssrlrni.du.q $vr12, $vr10, 4
# CHECK-ENCODING: encoding: [0x4c,0x11,0x56,0x73]
