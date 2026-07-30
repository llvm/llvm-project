# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vpickev.b $vr8, $vr13, $vr23
# CHECK-INST: vpickev.b $vr8, $vr13, $vr23
# CHECK-ENCODING: encoding: [0xa8,0x5d,0x1e,0x71]

vpickev.h $vr11, $vr18, $vr19
# CHECK-INST: vpickev.h $vr11, $vr18, $vr19
# CHECK-ENCODING: encoding: [0x4b,0xce,0x1e,0x71]

vpickev.w $vr16, $vr31, $vr30
# CHECK-INST: vpickev.w $vr16, $vr31, $vr30
# CHECK-ENCODING: encoding: [0xf0,0x7b,0x1f,0x71]

vpickev.d $vr1, $vr28, $vr8
# CHECK-INST: vpickev.d $vr1, $vr28, $vr8
# CHECK-ENCODING: encoding: [0x81,0xa3,0x1f,0x71]

vpickod.b $vr29, $vr28, $vr28
# CHECK-INST: vpickod.b $vr29, $vr28, $vr28
# CHECK-ENCODING: encoding: [0x9d,0x73,0x20,0x71]

vpickod.h $vr5, $vr5, $vr1
# CHECK-INST: vpickod.h $vr5, $vr5, $vr1
# CHECK-ENCODING: encoding: [0xa5,0x84,0x20,0x71]

vpickod.w $vr18, $vr8, $vr22
# CHECK-INST: vpickod.w $vr18, $vr8, $vr22
# CHECK-ENCODING: encoding: [0x12,0x59,0x21,0x71]

vpickod.d $vr5, $vr5, $vr22
# CHECK-INST: vpickod.d $vr5, $vr5, $vr22
# CHECK-ENCODING: encoding: [0xa5,0xd8,0x21,0x71]
