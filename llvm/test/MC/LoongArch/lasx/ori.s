# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvori.b $xr6, $xr2, 239
# CHECK-INST: xvori.b $xr6, $xr2, 239
# CHECK-ENCODING: encoding: [0x46,0xbc,0xd7,0x77]
