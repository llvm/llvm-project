# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vaddi.bu $vr14, $vr3, 2
# CHECK-INST: vaddi.bu $vr14, $vr3, 2
# CHECK-ENCODING: encoding: [0x6e,0x08,0x8a,0x72]

vaddi.hu $vr30, $vr27, 21
# CHECK-INST: vaddi.hu $vr30, $vr27, 21
# CHECK-ENCODING: encoding: [0x7e,0xd7,0x8a,0x72]

vaddi.wu $vr16, $vr28, 27
# CHECK-INST: vaddi.wu $vr16, $vr28, 27
# CHECK-ENCODING: encoding: [0x90,0x6f,0x8b,0x72]

vaddi.du $vr15, $vr8, 24
# CHECK-INST: vaddi.du $vr15, $vr8, 24
# CHECK-ENCODING: encoding: [0x0f,0xe1,0x8b,0x72]
