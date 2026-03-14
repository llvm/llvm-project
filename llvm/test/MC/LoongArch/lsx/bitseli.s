# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vbitseli.b $vr9, $vr0, 110
# CHECK-INST: vbitseli.b $vr9, $vr0, 110
# CHECK-ENCODING: encoding: [0x09,0xb8,0xc5,0x73]
