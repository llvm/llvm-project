# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvbitsel.v $xr18, $xr29, $xr15, $xr21
# CHECK-INST: xvbitsel.v $xr18, $xr29, $xr15, $xr21
# CHECK-ENCODING: encoding: [0xb2,0xbf,0x2a,0x0d]
