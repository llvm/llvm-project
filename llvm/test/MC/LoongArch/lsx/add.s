# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vadd.b $vr11, $vr12, $vr8
# CHECK-INST: vadd.b $vr11, $vr12, $vr8
# CHECK-ENCODING: encoding: [0x8b,0x21,0x0a,0x70]

vadd.h $vr22, $vr3, $vr4
# CHECK-INST: vadd.h $vr22, $vr3, $vr4
# CHECK-ENCODING: encoding: [0x76,0x90,0x0a,0x70]

vadd.w $vr13, $vr16, $vr6
# CHECK-INST: vadd.w $vr13, $vr16, $vr6
# CHECK-ENCODING: encoding: [0x0d,0x1a,0x0b,0x70]

vadd.d $vr12, $vr9, $vr3
# CHECK-INST: vadd.d $vr12, $vr9, $vr3
# CHECK-ENCODING: encoding: [0x2c,0x8d,0x0b,0x70]

vadd.q $vr16, $vr15, $vr10
# CHECK-INST: vadd.q $vr16, $vr15, $vr10
# CHECK-ENCODING: encoding: [0xf0,0x29,0x2d,0x71]
