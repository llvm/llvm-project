# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vbitrev.b $vr4, $vr31, $vr10
# CHECK-INST: vbitrev.b $vr4, $vr31, $vr10
# CHECK-ENCODING: encoding: [0xe4,0x2b,0x10,0x71]

vbitrev.h $vr19, $vr19, $vr16
# CHECK-INST: vbitrev.h $vr19, $vr19, $vr16
# CHECK-ENCODING: encoding: [0x73,0xc2,0x10,0x71]

vbitrev.w $vr4, $vr18, $vr7
# CHECK-INST: vbitrev.w $vr4, $vr18, $vr7
# CHECK-ENCODING: encoding: [0x44,0x1e,0x11,0x71]

vbitrev.d $vr17, $vr31, $vr0
# CHECK-INST: vbitrev.d $vr17, $vr31, $vr0
# CHECK-ENCODING: encoding: [0xf1,0x83,0x11,0x71]

vbitrevi.b $vr9, $vr31, 7
# CHECK-INST: vbitrevi.b $vr9, $vr31, 7
# CHECK-ENCODING: encoding: [0xe9,0x3f,0x18,0x73]

vbitrevi.h $vr4, $vr24, 8
# CHECK-INST: vbitrevi.h $vr4, $vr24, 8
# CHECK-ENCODING: encoding: [0x04,0x63,0x18,0x73]

vbitrevi.w $vr17, $vr19, 2
# CHECK-INST: vbitrevi.w $vr17, $vr19, 2
# CHECK-ENCODING: encoding: [0x71,0x8a,0x18,0x73]

vbitrevi.d $vr15, $vr7, 47
# CHECK-INST: vbitrevi.d $vr15, $vr7, 47
# CHECK-ENCODING: encoding: [0xef,0xbc,0x19,0x73]
