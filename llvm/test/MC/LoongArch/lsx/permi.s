# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vpermi.w $vr2, $vr22, 219
# CHECK-INST: vpermi.w $vr2, $vr22, 219
# CHECK-ENCODING: encoding: [0xc2,0x6e,0xe7,0x73]
