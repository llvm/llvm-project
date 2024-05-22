# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsubi.bu $vr21, $vr1, 16
# CHECK-INST: vsubi.bu $vr21, $vr1, 16
# CHECK-ENCODING: encoding: [0x35,0x40,0x8c,0x72]

vsubi.hu $vr10, $vr24, 8
# CHECK-INST: vsubi.hu $vr10, $vr24, 8
# CHECK-ENCODING: encoding: [0x0a,0xa3,0x8c,0x72]

vsubi.wu $vr10, $vr13, 8
# CHECK-INST: vsubi.wu $vr10, $vr13, 8
# CHECK-ENCODING: encoding: [0xaa,0x21,0x8d,0x72]

vsubi.du $vr27, $vr0, 29
# CHECK-INST: vsubi.du $vr27, $vr0, 29
# CHECK-ENCODING: encoding: [0x1b,0xf4,0x8d,0x72]
