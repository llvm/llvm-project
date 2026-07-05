# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsrlni.b.h $vr15, $vr25, 9
# CHECK-INST: vsrlni.b.h $vr15, $vr25, 9
# CHECK-ENCODING: encoding: [0x2f,0x67,0x40,0x73]

vsrlni.h.w $vr3, $vr0, 8
# CHECK-INST: vsrlni.h.w $vr3, $vr0, 8
# CHECK-ENCODING: encoding: [0x03,0xa0,0x40,0x73]

vsrlni.w.d $vr19, $vr26, 51
# CHECK-INST: vsrlni.w.d $vr19, $vr26, 51
# CHECK-ENCODING: encoding: [0x53,0xcf,0x41,0x73]

vsrlni.d.q $vr10, $vr18, 60
# CHECK-INST: vsrlni.d.q $vr10, $vr18, 60
# CHECK-ENCODING: encoding: [0x4a,0xf2,0x42,0x73]
