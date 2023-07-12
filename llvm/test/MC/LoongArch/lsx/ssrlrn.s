# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vssrlrn.b.h $vr28, $vr3, $vr15
# CHECK-INST: vssrlrn.b.h $vr28, $vr3, $vr15
# CHECK-ENCODING: encoding: [0x7c,0xbc,0x00,0x71]

vssrlrn.h.w $vr22, $vr0, $vr9
# CHECK-INST: vssrlrn.h.w $vr22, $vr0, $vr9
# CHECK-ENCODING: encoding: [0x16,0x24,0x01,0x71]

vssrlrn.w.d $vr6, $vr14, $vr21
# CHECK-INST: vssrlrn.w.d $vr6, $vr14, $vr21
# CHECK-ENCODING: encoding: [0xc6,0xd5,0x01,0x71]

vssrlrn.bu.h $vr10, $vr24, $vr12
# CHECK-INST: vssrlrn.bu.h $vr10, $vr24, $vr12
# CHECK-ENCODING: encoding: [0x0a,0xb3,0x08,0x71]

vssrlrn.hu.w $vr29, $vr6, $vr1
# CHECK-INST: vssrlrn.hu.w $vr29, $vr6, $vr1
# CHECK-ENCODING: encoding: [0xdd,0x04,0x09,0x71]

vssrlrn.wu.d $vr2, $vr23, $vr7
# CHECK-INST: vssrlrn.wu.d $vr2, $vr23, $vr7
# CHECK-ENCODING: encoding: [0xe2,0x9e,0x09,0x71]
