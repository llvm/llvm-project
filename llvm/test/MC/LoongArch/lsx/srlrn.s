# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsrlrn.b.h $vr10, $vr18, $vr11
# CHECK-INST: vsrlrn.b.h $vr10, $vr18, $vr11
# CHECK-ENCODING: encoding: [0x4a,0xae,0xf8,0x70]

vsrlrn.h.w $vr28, $vr15, $vr22
# CHECK-INST: vsrlrn.h.w $vr28, $vr15, $vr22
# CHECK-ENCODING: encoding: [0xfc,0x59,0xf9,0x70]

vsrlrn.w.d $vr19, $vr7, $vr26
# CHECK-INST: vsrlrn.w.d $vr19, $vr7, $vr26
# CHECK-ENCODING: encoding: [0xf3,0xe8,0xf9,0x70]
