# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vshuf4i.b $vr30, $vr14, 72
# CHECK-INST: vshuf4i.b $vr30, $vr14, 72
# CHECK-ENCODING: encoding: [0xde,0x21,0x91,0x73]

vshuf4i.h $vr13, $vr4, 222
# CHECK-INST: vshuf4i.h $vr13, $vr4, 222
# CHECK-ENCODING: encoding: [0x8d,0x78,0x97,0x73]

vshuf4i.w $vr17, $vr8, 74
# CHECK-INST: vshuf4i.w $vr17, $vr8, 74
# CHECK-ENCODING: encoding: [0x11,0x29,0x99,0x73]

vshuf4i.d $vr11, $vr6, 157
# CHECK-INST: vshuf4i.d $vr11, $vr6, 157
# CHECK-ENCODING: encoding: [0xcb,0x74,0x9e,0x73]
