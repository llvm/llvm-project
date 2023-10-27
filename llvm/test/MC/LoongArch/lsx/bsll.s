# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vbsll.v $vr21, $vr1, 17
# CHECK-INST: vbsll.v $vr21, $vr1, 17
# CHECK-ENCODING: encoding: [0x35,0x44,0x8e,0x72]
