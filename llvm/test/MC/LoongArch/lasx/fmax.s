# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfmax.s $xr29, $xr24, $xr8
# CHECK-INST: xvfmax.s $xr29, $xr24, $xr8
# CHECK-ENCODING: encoding: [0x1d,0xa3,0x3c,0x75]

xvfmax.d $xr31, $xr25, $xr23
# CHECK-INST: xvfmax.d $xr31, $xr25, $xr23
# CHECK-ENCODING: encoding: [0x3f,0x5f,0x3d,0x75]
