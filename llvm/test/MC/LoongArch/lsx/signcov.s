# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsigncov.b $vr11, $vr3, $vr7
# CHECK-INST: vsigncov.b $vr11, $vr3, $vr7
# CHECK-ENCODING: encoding: [0x6b,0x1c,0x2e,0x71]

vsigncov.h $vr8, $vr29, $vr1
# CHECK-INST: vsigncov.h $vr8, $vr29, $vr1
# CHECK-ENCODING: encoding: [0xa8,0x87,0x2e,0x71]

vsigncov.w $vr28, $vr13, $vr21
# CHECK-INST: vsigncov.w $vr28, $vr13, $vr21
# CHECK-ENCODING: encoding: [0xbc,0x55,0x2f,0x71]

vsigncov.d $vr22, $vr20, $vr0
# CHECK-INST: vsigncov.d $vr22, $vr20, $vr0
# CHECK-ENCODING: encoding: [0x96,0x82,0x2f,0x71]
