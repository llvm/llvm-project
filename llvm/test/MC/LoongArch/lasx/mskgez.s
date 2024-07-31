# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmskgez.b $xr30, $xr5
# CHECK-INST: xvmskgez.b $xr30, $xr5
# CHECK-ENCODING: encoding: [0xbe,0x50,0x9c,0x76]
