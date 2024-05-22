# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfdiv.s $xr29, $xr5, $xr12
# CHECK-INST: xvfdiv.s $xr29, $xr5, $xr12
# CHECK-ENCODING: encoding: [0xbd,0xb0,0x3a,0x75]

xvfdiv.d $xr31, $xr10, $xr30
# CHECK-INST: xvfdiv.d $xr31, $xr10, $xr30
# CHECK-ENCODING: encoding: [0x5f,0x79,0x3b,0x75]
