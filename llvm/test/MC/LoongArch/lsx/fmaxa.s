# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfmaxa.s $vr2, $vr8, $vr1
# CHECK-INST: vfmaxa.s $vr2, $vr8, $vr1
# CHECK-ENCODING: encoding: [0x02,0x85,0x40,0x71]

vfmaxa.d $vr1, $vr8, $vr28
# CHECK-INST: vfmaxa.d $vr1, $vr8, $vr28
# CHECK-ENCODING: encoding: [0x01,0x71,0x41,0x71]
