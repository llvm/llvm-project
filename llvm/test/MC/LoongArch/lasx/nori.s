# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvnori.b $xr7, $xr1, 209
# CHECK-INST: xvnori.b $xr7, $xr1, 209
# CHECK-ENCODING: encoding: [0x27,0x44,0xdf,0x77]
