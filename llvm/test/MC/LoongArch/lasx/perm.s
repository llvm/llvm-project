# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvperm.w $xr24, $xr23, $xr16
# CHECK-INST: xvperm.w $xr24, $xr23, $xr16
# CHECK-ENCODING: encoding: [0xf8,0x42,0x7d,0x75]
