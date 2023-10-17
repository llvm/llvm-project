# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfcvth.s.h $xr9, $xr25
# CHECK-INST: xvfcvth.s.h $xr9, $xr25
# CHECK-ENCODING: encoding: [0x29,0xef,0x9d,0x76]

xvfcvth.d.s $xr29, $xr17
# CHECK-INST: xvfcvth.d.s $xr29, $xr17
# CHECK-ENCODING: encoding: [0x3d,0xf6,0x9d,0x76]
