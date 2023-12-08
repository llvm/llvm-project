# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vnori.b $vr8, $vr16, 186
# CHECK-INST: vnori.b $vr8, $vr16, 186
# CHECK-ENCODING: encoding: [0x08,0xea,0xde,0x73]
