# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmuh.b $vr23, $vr18, $vr21
# CHECK-INST: vmuh.b $vr23, $vr18, $vr21
# CHECK-ENCODING: encoding: [0x57,0x56,0x86,0x70]

vmuh.h $vr25, $vr18, $vr5
# CHECK-INST: vmuh.h $vr25, $vr18, $vr5
# CHECK-ENCODING: encoding: [0x59,0x96,0x86,0x70]

vmuh.w $vr6, $vr9, $vr14
# CHECK-INST: vmuh.w $vr6, $vr9, $vr14
# CHECK-ENCODING: encoding: [0x26,0x39,0x87,0x70]

vmuh.d $vr31, $vr21, $vr8
# CHECK-INST: vmuh.d $vr31, $vr21, $vr8
# CHECK-ENCODING: encoding: [0xbf,0xa2,0x87,0x70]

vmuh.bu $vr11, $vr26, $vr7
# CHECK-INST: vmuh.bu $vr11, $vr26, $vr7
# CHECK-ENCODING: encoding: [0x4b,0x1f,0x88,0x70]

vmuh.hu $vr27, $vr4, $vr28
# CHECK-INST: vmuh.hu $vr27, $vr4, $vr28
# CHECK-ENCODING: encoding: [0x9b,0xf0,0x88,0x70]

vmuh.wu $vr28, $vr21, $vr28
# CHECK-INST: vmuh.wu $vr28, $vr21, $vr28
# CHECK-ENCODING: encoding: [0xbc,0x72,0x89,0x70]

vmuh.du $vr25, $vr3, $vr4
# CHECK-INST: vmuh.du $vr25, $vr3, $vr4
# CHECK-ENCODING: encoding: [0x79,0x90,0x89,0x70]
