# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vreplve.b $vr10, $vr31, $r20
# CHECK-INST: vreplve.b $vr10, $vr31, $t8
# CHECK-ENCODING: encoding: [0xea,0x53,0x22,0x71]

vreplve.h $vr8, $vr3, $r30
# CHECK-INST: vreplve.h $vr8, $vr3, $s7
# CHECK-ENCODING: encoding: [0x68,0xf8,0x22,0x71]

vreplve.w $vr5, $vr1, $r20
# CHECK-INST: vreplve.w $vr5, $vr1, $t8
# CHECK-ENCODING: encoding: [0x25,0x50,0x23,0x71]

vreplve.d $vr11, $vr15, $r30
# CHECK-INST: vreplve.d $vr11, $vr15, $s7
# CHECK-ENCODING: encoding: [0xeb,0xf9,0x23,0x71]
