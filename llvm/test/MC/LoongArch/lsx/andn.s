# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vandn.v $vr1, $vr26, $vr28
# CHECK-INST: vandn.v $vr1, $vr26, $vr28
# CHECK-ENCODING: encoding: [0x41,0x73,0x28,0x71]
