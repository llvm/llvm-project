# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vand.v $vr27, $vr30, $vr18
# CHECK-INST: vand.v $vr27, $vr30, $vr18
# CHECK-ENCODING: encoding: [0xdb,0x4b,0x26,0x71]
