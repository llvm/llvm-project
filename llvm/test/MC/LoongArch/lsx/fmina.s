# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfmina.s $vr20, $vr27, $vr20
# CHECK-INST: vfmina.s $vr20, $vr27, $vr20
# CHECK-ENCODING: encoding: [0x74,0xd3,0x42,0x71]

vfmina.d $vr1, $vr26, $vr22
# CHECK-INST: vfmina.d $vr1, $vr26, $vr22
# CHECK-ENCODING: encoding: [0x41,0x5b,0x43,0x71]
