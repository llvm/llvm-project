# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vxori.b $vr13, $vr4, 74
# CHECK-INST: vxori.b $vr13, $vr4, 74
# CHECK-ENCODING: encoding: [0x8d,0x28,0xd9,0x73]
