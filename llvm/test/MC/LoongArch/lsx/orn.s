# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vorn.v $vr11, $vr20, $vr17
# CHECK-INST: vorn.v $vr11, $vr20, $vr17
# CHECK-ENCODING: encoding: [0x8b,0xc6,0x28,0x71]
