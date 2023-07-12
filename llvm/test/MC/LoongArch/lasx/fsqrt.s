# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfsqrt.s $xr4, $xr27
# CHECK-INST: xvfsqrt.s $xr4, $xr27
# CHECK-ENCODING: encoding: [0x64,0xe7,0x9c,0x76]

xvfsqrt.d $xr26, $xr2
# CHECK-INST: xvfsqrt.d $xr26, $xr2
# CHECK-ENCODING: encoding: [0x5a,0xe8,0x9c,0x76]
