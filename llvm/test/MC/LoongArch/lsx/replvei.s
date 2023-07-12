# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vreplvei.b $vr23, $vr3, 3
# CHECK-INST: vreplvei.b $vr23, $vr3, 3
# CHECK-ENCODING: encoding: [0x77,0x8c,0xf7,0x72]

vreplvei.h $vr27, $vr16, 0
# CHECK-INST: vreplvei.h $vr27, $vr16, 0
# CHECK-ENCODING: encoding: [0x1b,0xc2,0xf7,0x72]

vreplvei.w $vr18, $vr23, 3
# CHECK-INST: vreplvei.w $vr18, $vr23, 3
# CHECK-ENCODING: encoding: [0xf2,0xee,0xf7,0x72]

vreplvei.d $vr15, $vr12, 1
# CHECK-INST: vreplvei.d $vr15, $vr12, 1
# CHECK-ENCODING: encoding: [0x8f,0xf5,0xf7,0x72]
