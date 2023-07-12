# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfsub.s $vr4, $vr9, $vr12
# CHECK-INST: vfsub.s $vr4, $vr9, $vr12
# CHECK-ENCODING: encoding: [0x24,0xb1,0x32,0x71]

vfsub.d $vr12, $vr28, $vr27
# CHECK-INST: vfsub.d $vr12, $vr28, $vr27
# CHECK-ENCODING: encoding: [0x8c,0x6f,0x33,0x71]
