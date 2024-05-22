# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvpcnt.b $xr8, $xr27
# CHECK-INST: xvpcnt.b $xr8, $xr27
# CHECK-ENCODING: encoding: [0x68,0x23,0x9c,0x76]

xvpcnt.h $xr12, $xr4
# CHECK-INST: xvpcnt.h $xr12, $xr4
# CHECK-ENCODING: encoding: [0x8c,0x24,0x9c,0x76]

xvpcnt.w $xr31, $xr23
# CHECK-INST: xvpcnt.w $xr31, $xr23
# CHECK-ENCODING: encoding: [0xff,0x2a,0x9c,0x76]

xvpcnt.d $xr26, $xr12
# CHECK-INST: xvpcnt.d $xr26, $xr12
# CHECK-ENCODING: encoding: [0x9a,0x2d,0x9c,0x76]
