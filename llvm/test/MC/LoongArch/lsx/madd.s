# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmadd.b $vr13, $vr5, $vr10
# CHECK-INST: vmadd.b $vr13, $vr5, $vr10
# CHECK-ENCODING: encoding: [0xad,0x28,0xa8,0x70]

vmadd.h $vr11, $vr15, $vr8
# CHECK-INST: vmadd.h $vr11, $vr15, $vr8
# CHECK-ENCODING: encoding: [0xeb,0xa1,0xa8,0x70]

vmadd.w $vr5, $vr17, $vr16
# CHECK-INST: vmadd.w $vr5, $vr17, $vr16
# CHECK-ENCODING: encoding: [0x25,0x42,0xa9,0x70]

vmadd.d $vr29, $vr11, $vr12
# CHECK-INST: vmadd.d $vr29, $vr11, $vr12
# CHECK-ENCODING: encoding: [0x7d,0xb1,0xa9,0x70]
