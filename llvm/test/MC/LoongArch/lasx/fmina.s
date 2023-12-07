# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfmina.s $xr29, $xr27, $xr17
# CHECK-INST: xvfmina.s $xr29, $xr27, $xr17
# CHECK-ENCODING: encoding: [0x7d,0xc7,0x42,0x75]

xvfmina.d $xr12, $xr20, $xr18
# CHECK-INST: xvfmina.d $xr12, $xr20, $xr18
# CHECK-ENCODING: encoding: [0x8c,0x4a,0x43,0x75]
