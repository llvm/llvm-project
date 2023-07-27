# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmsub.b $vr19, $vr20, $vr12
# CHECK-INST: vmsub.b $vr19, $vr20, $vr12
# CHECK-ENCODING: encoding: [0x93,0x32,0xaa,0x70]

vmsub.h $vr1, $vr9, $vr22
# CHECK-INST: vmsub.h $vr1, $vr9, $vr22
# CHECK-ENCODING: encoding: [0x21,0xd9,0xaa,0x70]

vmsub.w $vr10, $vr2, $vr13
# CHECK-INST: vmsub.w $vr10, $vr2, $vr13
# CHECK-ENCODING: encoding: [0x4a,0x34,0xab,0x70]

vmsub.d $vr28, $vr31, $vr6
# CHECK-INST: vmsub.d $vr28, $vr31, $vr6
# CHECK-ENCODING: encoding: [0xfc,0x9b,0xab,0x70]
