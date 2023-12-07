# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfmax.s $vr19, $vr25, $vr16
# CHECK-INST: vfmax.s $vr19, $vr25, $vr16
# CHECK-ENCODING: encoding: [0x33,0xc3,0x3c,0x71]

vfmax.d $vr19, $vr21, $vr12
# CHECK-INST: vfmax.d $vr19, $vr21, $vr12
# CHECK-ENCODING: encoding: [0xb3,0x32,0x3d,0x71]
