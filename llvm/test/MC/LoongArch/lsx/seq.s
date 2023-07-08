# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vseq.b $vr15, $vr30, $vr24
# CHECK-INST: vseq.b $vr15, $vr30, $vr24
# CHECK-ENCODING: encoding: [0xcf,0x63,0x00,0x70]

vseq.h $vr7, $vr4, $vr22
# CHECK-INST: vseq.h $vr7, $vr4, $vr22
# CHECK-ENCODING: encoding: [0x87,0xd8,0x00,0x70]

vseq.w $vr4, $vr15, $vr28
# CHECK-INST: vseq.w $vr4, $vr15, $vr28
# CHECK-ENCODING: encoding: [0xe4,0x71,0x01,0x70]

vseq.d $vr29, $vr26, $vr22
# CHECK-INST: vseq.d $vr29, $vr26, $vr22
# CHECK-ENCODING: encoding: [0x5d,0xdb,0x01,0x70]

vseqi.b $vr19, $vr30, 14
# CHECK-INST: vseqi.b $vr19, $vr30, 14
# CHECK-ENCODING: encoding: [0xd3,0x3b,0x80,0x72]

vseqi.h $vr15, $vr2, 15
# CHECK-INST: vseqi.h $vr15, $vr2, 15
# CHECK-ENCODING: encoding: [0x4f,0xbc,0x80,0x72]

vseqi.w $vr27, $vr23, -10
# CHECK-INST: vseqi.w $vr27, $vr23, -10
# CHECK-ENCODING: encoding: [0xfb,0x5a,0x81,0x72]

vseqi.d $vr6, $vr12, -2
# CHECK-INST: vseqi.d $vr6, $vr12, -2
# CHECK-ENCODING: encoding: [0x86,0xf9,0x81,0x72]
