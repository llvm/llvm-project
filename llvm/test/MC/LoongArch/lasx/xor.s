# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvxor.v $xr14, $xr26, $xr10
# CHECK-INST: xvxor.v $xr14, $xr26, $xr10
# CHECK-ENCODING: encoding: [0x4e,0x2b,0x27,0x75]
