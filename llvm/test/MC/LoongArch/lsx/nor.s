# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vnor.v $vr18, $vr5, $vr29
# CHECK-INST: vnor.v $vr18, $vr5, $vr29
# CHECK-ENCODING: encoding: [0xb2,0xf4,0x27,0x71]
