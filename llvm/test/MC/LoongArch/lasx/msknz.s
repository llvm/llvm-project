# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmsknz.b $xr22, $xr22
# CHECK-INST: xvmsknz.b $xr22, $xr22
# CHECK-ENCODING: encoding: [0xd6,0x62,0x9c,0x76]
