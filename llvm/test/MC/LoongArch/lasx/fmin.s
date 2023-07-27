# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfmin.s $xr31, $xr5, $xr16
# CHECK-INST: xvfmin.s $xr31, $xr5, $xr16
# CHECK-ENCODING: encoding: [0xbf,0xc0,0x3e,0x75]

xvfmin.d $xr13, $xr30, $xr25
# CHECK-INST: xvfmin.d $xr13, $xr30, $xr25
# CHECK-ENCODING: encoding: [0xcd,0x67,0x3f,0x75]
