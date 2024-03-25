# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvorn.v $xr17, $xr29, $xr5
# CHECK-INST: xvorn.v $xr17, $xr29, $xr5
# CHECK-ENCODING: encoding: [0xb1,0x97,0x28,0x75]
