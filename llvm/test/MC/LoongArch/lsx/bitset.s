# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vbitset.b $vr13, $vr27, $vr14
# CHECK-INST: vbitset.b $vr13, $vr27, $vr14
# CHECK-ENCODING: encoding: [0x6d,0x3b,0x0e,0x71]

vbitset.h $vr24, $vr6, $vr3
# CHECK-INST: vbitset.h $vr24, $vr6, $vr3
# CHECK-ENCODING: encoding: [0xd8,0x8c,0x0e,0x71]

vbitset.w $vr31, $vr0, $vr0
# CHECK-INST: vbitset.w $vr31, $vr0, $vr0
# CHECK-ENCODING: encoding: [0x1f,0x00,0x0f,0x71]

vbitset.d $vr6, $vr15, $vr31
# CHECK-INST: vbitset.d $vr6, $vr15, $vr31
# CHECK-ENCODING: encoding: [0xe6,0xfd,0x0f,0x71]

vbitseti.b $vr4, $vr3, 1
# CHECK-INST: vbitseti.b $vr4, $vr3, 1
# CHECK-ENCODING: encoding: [0x64,0x24,0x14,0x73]

vbitseti.h $vr10, $vr20, 2
# CHECK-INST: vbitseti.h $vr10, $vr20, 2
# CHECK-ENCODING: encoding: [0x8a,0x4a,0x14,0x73]

vbitseti.w $vr14, $vr16, 4
# CHECK-INST: vbitseti.w $vr14, $vr16, 4
# CHECK-ENCODING: encoding: [0x0e,0x92,0x14,0x73]

vbitseti.d $vr10, $vr13, 25
# CHECK-INST: vbitseti.d $vr10, $vr13, 25
# CHECK-ENCODING: encoding: [0xaa,0x65,0x15,0x73]
