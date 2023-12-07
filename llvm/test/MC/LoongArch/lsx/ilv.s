# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vilvl.b $vr9, $vr30, $vr20
# CHECK-INST: vilvl.b $vr9, $vr30, $vr20
# CHECK-ENCODING: encoding: [0xc9,0x53,0x1a,0x71]

vilvl.h $vr6, $vr19, $vr30
# CHECK-INST: vilvl.h $vr6, $vr19, $vr30
# CHECK-ENCODING: encoding: [0x66,0xfa,0x1a,0x71]

vilvl.w $vr18, $vr3, $vr15
# CHECK-INST: vilvl.w $vr18, $vr3, $vr15
# CHECK-ENCODING: encoding: [0x72,0x3c,0x1b,0x71]

vilvl.d $vr20, $vr22, $vr9
# CHECK-INST: vilvl.d $vr20, $vr22, $vr9
# CHECK-ENCODING: encoding: [0xd4,0xa6,0x1b,0x71]

vilvh.b $vr14, $vr4, $vr12
# CHECK-INST: vilvh.b $vr14, $vr4, $vr12
# CHECK-ENCODING: encoding: [0x8e,0x30,0x1c,0x71]

vilvh.h $vr2, $vr0, $vr6
# CHECK-INST: vilvh.h $vr2, $vr0, $vr6
# CHECK-ENCODING: encoding: [0x02,0x98,0x1c,0x71]

vilvh.w $vr7, $vr27, $vr15
# CHECK-INST: vilvh.w $vr7, $vr27, $vr15
# CHECK-ENCODING: encoding: [0x67,0x3f,0x1d,0x71]

vilvh.d $vr9, $vr25, $vr29
# CHECK-INST: vilvh.d $vr9, $vr25, $vr29
# CHECK-ENCODING: encoding: [0x29,0xf7,0x1d,0x71]
