# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfrecip.s $xr3, $xr16
# CHECK-INST: xvfrecip.s $xr3, $xr16
# CHECK-ENCODING: encoding: [0x03,0xf6,0x9c,0x76]

xvfrecip.d $xr17, $xr24
# CHECK-INST: xvfrecip.d $xr17, $xr24
# CHECK-ENCODING: encoding: [0x11,0xfb,0x9c,0x76]
