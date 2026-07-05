# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vextrins.b $vr14, $vr19, 213
# CHECK-INST: vextrins.b $vr14, $vr19, 213
# CHECK-ENCODING: encoding: [0x6e,0x56,0x8f,0x73]

vextrins.h $vr1, $vr6, 170
# CHECK-INST: vextrins.h $vr1, $vr6, 170
# CHECK-ENCODING: encoding: [0xc1,0xa8,0x8a,0x73]

vextrins.w $vr9, $vr4, 189
# CHECK-INST: vextrins.w $vr9, $vr4, 189
# CHECK-ENCODING: encoding: [0x89,0xf4,0x86,0x73]

vextrins.d $vr20, $vr25, 121
# CHECK-INST: vextrins.d $vr20, $vr25, 121
# CHECK-ENCODING: encoding: [0x34,0xe7,0x81,0x73]
