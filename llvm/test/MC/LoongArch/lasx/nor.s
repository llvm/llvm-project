# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvnor.v $xr4, $xr23, $xr3
# CHECK-INST: xvnor.v $xr4, $xr23, $xr3
# CHECK-ENCODING: encoding: [0xe4,0x8e,0x27,0x75]
