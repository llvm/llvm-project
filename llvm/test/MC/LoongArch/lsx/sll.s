# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsll.b $vr31, $vr13, $vr5
# CHECK-INST: vsll.b $vr31, $vr13, $vr5
# CHECK-ENCODING: encoding: [0xbf,0x15,0xe8,0x70]

vsll.h $vr31, $vr1, $vr4
# CHECK-INST: vsll.h $vr31, $vr1, $vr4
# CHECK-ENCODING: encoding: [0x3f,0x90,0xe8,0x70]

vsll.w $vr8, $vr19, $vr19
# CHECK-INST: vsll.w $vr8, $vr19, $vr19
# CHECK-ENCODING: encoding: [0x68,0x4e,0xe9,0x70]

vsll.d $vr6, $vr25, $vr6
# CHECK-INST: vsll.d $vr6, $vr25, $vr6
# CHECK-ENCODING: encoding: [0x26,0x9b,0xe9,0x70]

vslli.b $vr6, $vr7, 2
# CHECK-INST: vslli.b $vr6, $vr7, 2
# CHECK-ENCODING: encoding: [0xe6,0x28,0x2c,0x73]

vslli.h $vr6, $vr4, 10
# CHECK-INST: vslli.h $vr6, $vr4, 10
# CHECK-ENCODING: encoding: [0x86,0x68,0x2c,0x73]

vslli.w $vr3, $vr13, 17
# CHECK-INST: vslli.w $vr3, $vr13, 17
# CHECK-ENCODING: encoding: [0xa3,0xc5,0x2c,0x73]

vslli.d $vr24, $vr11, 38
# CHECK-INST: vslli.d $vr24, $vr11, 38
# CHECK-ENCODING: encoding: [0x78,0x99,0x2d,0x73]
