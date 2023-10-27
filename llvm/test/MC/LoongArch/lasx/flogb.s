# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvflogb.s $xr17, $xr12
# CHECK-INST: xvflogb.s $xr17, $xr12
# CHECK-ENCODING: encoding: [0x91,0xc5,0x9c,0x76]

xvflogb.d $xr26, $xr1
# CHECK-INST: xvflogb.d $xr26, $xr1
# CHECK-ENCODING: encoding: [0x3a,0xc8,0x9c,0x76]
