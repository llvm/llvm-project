# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvxori.b $xr26, $xr8, 149
# CHECK-INST: xvxori.b $xr26, $xr8, 149
# CHECK-ENCODING: encoding: [0x1a,0x55,0xda,0x77]
