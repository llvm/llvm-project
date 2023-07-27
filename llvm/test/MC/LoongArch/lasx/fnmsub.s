# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfnmsub.s $xr22, $xr5, $xr4, $xr11
# CHECK-INST: xvfnmsub.s $xr22, $xr5, $xr4, $xr11
# CHECK-ENCODING: encoding: [0xb6,0x90,0xd5,0x0a]

xvfnmsub.d $xr8, $xr0, $xr29, $xr28
# CHECK-INST: xvfnmsub.d $xr8, $xr0, $xr29, $xr28
# CHECK-ENCODING: encoding: [0x08,0x74,0xee,0x0a]
