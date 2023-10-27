# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvbitseli.b $xr13, $xr21, 121
# CHECK-INST: xvbitseli.b $xr13, $xr21, 121
# CHECK-ENCODING: encoding: [0xad,0xe6,0xc5,0x77]
