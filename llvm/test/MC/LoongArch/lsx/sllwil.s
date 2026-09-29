# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsllwil.h.b $vr7, $vr6, 3
# CHECK-INST: vsllwil.h.b $vr7, $vr6, 3
# CHECK-ENCODING: encoding: [0xc7,0x2c,0x08,0x73]

vsllwil.w.h $vr6, $vr5, 8
# CHECK-INST: vsllwil.w.h $vr6, $vr5, 8
# CHECK-ENCODING: encoding: [0xa6,0x60,0x08,0x73]

vsllwil.d.w $vr15, $vr1, 22
# CHECK-INST: vsllwil.d.w $vr15, $vr1, 22
# CHECK-ENCODING: encoding: [0x2f,0xd8,0x08,0x73]

vsllwil.hu.bu $vr13, $vr4, 4
# CHECK-INST: vsllwil.hu.bu $vr13, $vr4, 4
# CHECK-ENCODING: encoding: [0x8d,0x30,0x0c,0x73]

vsllwil.wu.hu $vr1, $vr4, 3
# CHECK-INST: vsllwil.wu.hu $vr1, $vr4, 3
# CHECK-ENCODING: encoding: [0x81,0x4c,0x0c,0x73]

vsllwil.du.wu $vr18, $vr29, 25
# CHECK-INST: vsllwil.du.wu $vr18, $vr29, 25
# CHECK-ENCODING: encoding: [0xb2,0xe7,0x0c,0x73]
