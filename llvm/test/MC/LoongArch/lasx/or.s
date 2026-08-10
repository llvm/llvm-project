# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvor.v $xr6, $xr29, $xr21
# CHECK-INST: xvor.v $xr6, $xr29, $xr21
# CHECK-ENCODING: encoding: [0xa6,0xd7,0x26,0x75]
