# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvbsrl.v $xr4, $xr5, 29
# CHECK-INST: xvbsrl.v $xr4, $xr5, 29
# CHECK-ENCODING: encoding: [0xa4,0xf4,0x8e,0x76]
