# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvandn.v $xr3, $xr15, $xr3
# CHECK-INST: xvandn.v $xr3, $xr15, $xr3
# CHECK-ENCODING: encoding: [0xe3,0x0d,0x28,0x75]
