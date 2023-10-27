# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvand.v $xr14, $xr23, $xr19
# CHECK-INST: xvand.v $xr14, $xr23, $xr19
# CHECK-ENCODING: encoding: [0xee,0x4e,0x26,0x75]
