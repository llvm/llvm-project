# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vori.b $vr23, $vr3, 252
# CHECK-INST: vori.b $vr23, $vr3, 252
# CHECK-ENCODING: encoding: [0x77,0xf0,0xd7,0x73]
