# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfmsub.s $xr17, $xr3, $xr3, $xr23
# CHECK-INST: xvfmsub.s $xr17, $xr3, $xr3, $xr23
# CHECK-ENCODING: encoding: [0x71,0x8c,0x5b,0x0a]

xvfmsub.d $xr30, $xr15, $xr16, $xr14
# CHECK-INST: xvfmsub.d $xr30, $xr15, $xr16, $xr14
# CHECK-ENCODING: encoding: [0xfe,0x41,0x67,0x0a]
