# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vadda.b $vr7, $vr14, $vr21
# CHECK-INST: vadda.b $vr7, $vr14, $vr21
# CHECK-ENCODING: encoding: [0xc7,0x55,0x5c,0x70]

vadda.h $vr19, $vr29, $vr2
# CHECK-INST: vadda.h $vr19, $vr29, $vr2
# CHECK-ENCODING: encoding: [0xb3,0x8b,0x5c,0x70]

vadda.w $vr2, $vr23, $vr17
# CHECK-INST: vadda.w $vr2, $vr23, $vr17
# CHECK-ENCODING: encoding: [0xe2,0x46,0x5d,0x70]

vadda.d $vr13, $vr18, $vr24
# CHECK-INST: vadda.d $vr13, $vr18, $vr24
# CHECK-ENCODING: encoding: [0x4d,0xe2,0x5d,0x70]
