# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfadd.s $vr10, $vr2, $vr15
# CHECK-INST: vfadd.s $vr10, $vr2, $vr15
# CHECK-ENCODING: encoding: [0x4a,0xbc,0x30,0x71]

vfadd.d $vr16, $vr1, $vr22
# CHECK-INST: vfadd.d $vr16, $vr1, $vr22
# CHECK-ENCODING: encoding: [0x30,0x58,0x31,0x71]
