# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsat.b $vr29, $vr0, 1
# CHECK-INST: vsat.b $vr29, $vr0, 1
# CHECK-ENCODING: encoding: [0x1d,0x24,0x24,0x73]

vsat.h $vr4, $vr13, 13
# CHECK-INST: vsat.h $vr4, $vr13, 13
# CHECK-ENCODING: encoding: [0xa4,0x75,0x24,0x73]

vsat.w $vr6, $vr29, 19
# CHECK-INST: vsat.w $vr6, $vr29, 19
# CHECK-ENCODING: encoding: [0xa6,0xcf,0x24,0x73]

vsat.d $vr22, $vr6, 54
# CHECK-INST: vsat.d $vr22, $vr6, 54
# CHECK-ENCODING: encoding: [0xd6,0xd8,0x25,0x73]

vsat.bu $vr17, $vr8, 6
# CHECK-INST: vsat.bu $vr17, $vr8, 6
# CHECK-ENCODING: encoding: [0x11,0x39,0x28,0x73]

vsat.hu $vr2, $vr14, 2
# CHECK-INST: vsat.hu $vr2, $vr14, 2
# CHECK-ENCODING: encoding: [0xc2,0x49,0x28,0x73]

vsat.wu $vr1, $vr28, 19
# CHECK-INST: vsat.wu $vr1, $vr28, 19
# CHECK-ENCODING: encoding: [0x81,0xcf,0x28,0x73]

vsat.du $vr25, $vr6, 59
# CHECK-INST: vsat.du $vr25, $vr6, 59
# CHECK-ENCODING: encoding: [0xd9,0xec,0x29,0x73]
