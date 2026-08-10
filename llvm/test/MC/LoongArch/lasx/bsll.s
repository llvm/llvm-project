# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvbsll.v $xr14, $xr21, 20
# CHECK-INST: xvbsll.v $xr14, $xr21, 20
# CHECK-ENCODING: encoding: [0xae,0x52,0x8e,0x76]
