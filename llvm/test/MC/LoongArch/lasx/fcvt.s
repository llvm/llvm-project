# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfcvt.h.s $xr9, $xr17, $xr23
# CHECK-INST: xvfcvt.h.s $xr9, $xr17, $xr23
# CHECK-ENCODING: encoding: [0x29,0x5e,0x46,0x75]

xvfcvt.s.d $xr27, $xr10, $xr29
# CHECK-INST: xvfcvt.s.d $xr27, $xr10, $xr29
# CHECK-ENCODING: encoding: [0x5b,0xf5,0x46,0x75]
