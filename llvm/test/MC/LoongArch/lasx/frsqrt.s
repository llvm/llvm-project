# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfrsqrt.s $xr31, $xr25
# CHECK-INST: xvfrsqrt.s $xr31, $xr25
# CHECK-ENCODING: encoding: [0x3f,0x07,0x9d,0x76]

xvfrsqrt.d $xr14, $xr22
# CHECK-INST: xvfrsqrt.d $xr14, $xr22
# CHECK-ENCODING: encoding: [0xce,0x0a,0x9d,0x76]
