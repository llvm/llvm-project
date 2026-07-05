# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfmadd.s $vr6, $vr7, $vr13, $vr24
# CHECK-INST: vfmadd.s $vr6, $vr7, $vr13, $vr24
# CHECK-ENCODING: encoding: [0xe6,0x34,0x1c,0x09]

vfmadd.d $vr3, $vr28, $vr2, $vr21
# CHECK-INST: vfmadd.d $vr3, $vr28, $vr2, $vr21
# CHECK-ENCODING: encoding: [0x83,0x8b,0x2a,0x09]
