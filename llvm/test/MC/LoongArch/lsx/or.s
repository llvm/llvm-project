# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vor.v $vr9, $vr18, $vr24
# CHECK-INST: vor.v $vr9, $vr18, $vr24
# CHECK-ENCODING: encoding: [0x49,0xe2,0x26,0x71]
