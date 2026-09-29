# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vbitsel.v $vr2, $vr28, $vr6, $vr30
# CHECK-INST: vbitsel.v $vr2, $vr28, $vr6, $vr30
# CHECK-ENCODING: encoding: [0x82,0x1b,0x1f,0x0d]
