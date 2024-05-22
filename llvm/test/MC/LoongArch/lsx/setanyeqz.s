# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsetanyeqz.b $fcc3, $vr4
# CHECK-INST: vsetanyeqz.b $fcc3, $vr4
# CHECK-ENCODING: encoding: [0x83,0xa0,0x9c,0x72]

vsetanyeqz.h $fcc2, $vr15
# CHECK-INST: vsetanyeqz.h $fcc2, $vr15
# CHECK-ENCODING: encoding: [0xe2,0xa5,0x9c,0x72]

vsetanyeqz.w $fcc4, $vr0
# CHECK-INST: vsetanyeqz.w $fcc4, $vr0
# CHECK-ENCODING: encoding: [0x04,0xa8,0x9c,0x72]

vsetanyeqz.d $fcc3, $vr7
# CHECK-INST: vsetanyeqz.d $fcc3, $vr7
# CHECK-ENCODING: encoding: [0xe3,0xac,0x9c,0x72]
