# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vpackev.b $vr1, $vr27, $vr16
# CHECK-INST: vpackev.b $vr1, $vr27, $vr16
# CHECK-ENCODING: encoding: [0x61,0x43,0x16,0x71]

vpackev.h $vr0, $vr3, $vr25
# CHECK-INST: vpackev.h $vr0, $vr3, $vr25
# CHECK-ENCODING: encoding: [0x60,0xe4,0x16,0x71]

vpackev.w $vr10, $vr4, $vr29
# CHECK-INST: vpackev.w $vr10, $vr4, $vr29
# CHECK-ENCODING: encoding: [0x8a,0x74,0x17,0x71]

vpackev.d $vr28, $vr6, $vr7
# CHECK-INST: vpackev.d $vr28, $vr6, $vr7
# CHECK-ENCODING: encoding: [0xdc,0x9c,0x17,0x71]

vpackod.b $vr14, $vr13, $vr7
# CHECK-INST: vpackod.b $vr14, $vr13, $vr7
# CHECK-ENCODING: encoding: [0xae,0x1d,0x18,0x71]

vpackod.h $vr28, $vr5, $vr7
# CHECK-INST: vpackod.h $vr28, $vr5, $vr7
# CHECK-ENCODING: encoding: [0xbc,0x9c,0x18,0x71]

vpackod.w $vr15, $vr11, $vr17
# CHECK-INST: vpackod.w $vr15, $vr11, $vr17
# CHECK-ENCODING: encoding: [0x6f,0x45,0x19,0x71]

vpackod.d $vr12, $vr15, $vr0
# CHECK-INST: vpackod.d $vr12, $vr15, $vr0
# CHECK-ENCODING: encoding: [0xec,0x81,0x19,0x71]
