# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vbsrl.v $vr14, $vr15, 24
# CHECK-INST: vbsrl.v $vr14, $vr15, 24
# CHECK-ENCODING: encoding: [0xee,0xe1,0x8e,0x72]
