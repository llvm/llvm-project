# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfmaxa.s $xr15, $xr18, $xr5
# CHECK-INST: xvfmaxa.s $xr15, $xr18, $xr5
# CHECK-ENCODING: encoding: [0x4f,0x96,0x40,0x75]

xvfmaxa.d $xr2, $xr20, $xr29
# CHECK-INST: xvfmaxa.d $xr2, $xr20, $xr29
# CHECK-ENCODING: encoding: [0x82,0x76,0x41,0x75]
