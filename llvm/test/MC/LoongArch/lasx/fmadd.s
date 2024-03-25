# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfmadd.s $xr5, $xr31, $xr31, $xr27
# CHECK-INST: xvfmadd.s $xr5, $xr31, $xr31, $xr27
# CHECK-ENCODING: encoding: [0xe5,0xff,0x1d,0x0a]

xvfmadd.d $xr9, $xr16, $xr31, $xr25
# CHECK-INST: xvfmadd.d $xr9, $xr16, $xr31, $xr25
# CHECK-ENCODING: encoding: [0x09,0xfe,0x2c,0x0a]
