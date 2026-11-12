# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfdiv.s $vr27, $vr12, $vr26
# CHECK-INST: vfdiv.s $vr27, $vr12, $vr26
# CHECK-ENCODING: encoding: [0x9b,0xe9,0x3a,0x71]

vfdiv.d $vr3, $vr1, $vr7
# CHECK-INST: vfdiv.d $vr3, $vr1, $vr7
# CHECK-ENCODING: encoding: [0x23,0x1c,0x3b,0x71]
