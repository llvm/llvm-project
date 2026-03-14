# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vslt.b $vr24, $vr23, $vr26
# CHECK-INST: vslt.b $vr24, $vr23, $vr26
# CHECK-ENCODING: encoding: [0xf8,0x6a,0x06,0x70]

vslt.h $vr23, $vr4, $vr6
# CHECK-INST: vslt.h $vr23, $vr4, $vr6
# CHECK-ENCODING: encoding: [0x97,0x98,0x06,0x70]

vslt.w $vr30, $vr25, $vr1
# CHECK-INST: vslt.w $vr30, $vr25, $vr1
# CHECK-ENCODING: encoding: [0x3e,0x07,0x07,0x70]

vslt.d $vr25, $vr22, $vr15
# CHECK-INST: vslt.d $vr25, $vr22, $vr15
# CHECK-ENCODING: encoding: [0xd9,0xbe,0x07,0x70]

vslti.b $vr11, $vr12, -10
# CHECK-INST: vslti.b $vr11, $vr12, -10
# CHECK-ENCODING: encoding: [0x8b,0x59,0x86,0x72]

vslti.h $vr20, $vr12, -8
# CHECK-INST: vslti.h $vr20, $vr12, -8
# CHECK-ENCODING: encoding: [0x94,0xe1,0x86,0x72]

vslti.w $vr20, $vr27, 0
# CHECK-INST: vslti.w $vr20, $vr27, 0
# CHECK-ENCODING: encoding: [0x74,0x03,0x87,0x72]

vslti.d $vr19, $vr18, 4
# CHECK-INST: vslti.d $vr19, $vr18, 4
# CHECK-ENCODING: encoding: [0x53,0x92,0x87,0x72]

vslt.bu $vr5, $vr30, $vr28
# CHECK-INST: vslt.bu $vr5, $vr30, $vr28
# CHECK-ENCODING: encoding: [0xc5,0x73,0x08,0x70]

vslt.hu $vr13, $vr28, $vr23
# CHECK-INST: vslt.hu $vr13, $vr28, $vr23
# CHECK-ENCODING: encoding: [0x8d,0xdf,0x08,0x70]

vslt.wu $vr20, $vr28, $vr1
# CHECK-INST: vslt.wu $vr20, $vr28, $vr1
# CHECK-ENCODING: encoding: [0x94,0x07,0x09,0x70]

vslt.du $vr6, $vr6, $vr5
# CHECK-INST: vslt.du $vr6, $vr6, $vr5
# CHECK-ENCODING: encoding: [0xc6,0x94,0x09,0x70]

vslti.bu $vr9, $vr29, 23
# CHECK-INST: vslti.bu $vr9, $vr29, 23
# CHECK-ENCODING: encoding: [0xa9,0x5f,0x88,0x72]

vslti.hu $vr28, $vr13, 6
# CHECK-INST: vslti.hu $vr28, $vr13, 6
# CHECK-ENCODING: encoding: [0xbc,0x99,0x88,0x72]

vslti.wu $vr11, $vr9, 12
# CHECK-INST: vslti.wu $vr11, $vr9, 12
# CHECK-ENCODING: encoding: [0x2b,0x31,0x89,0x72]

vslti.du $vr23, $vr30, 21
# CHECK-INST: vslti.du $vr23, $vr30, 21
# CHECK-ENCODING: encoding: [0xd7,0xd7,0x89,0x72]
