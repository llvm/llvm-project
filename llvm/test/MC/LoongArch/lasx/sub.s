# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsub.b $xr11, $xr28, $xr16
# CHECK-INST: xvsub.b $xr11, $xr28, $xr16
# CHECK-ENCODING: encoding: [0x8b,0x43,0x0c,0x74]

xvsub.h $xr11, $xr3, $xr24
# CHECK-INST: xvsub.h $xr11, $xr3, $xr24
# CHECK-ENCODING: encoding: [0x6b,0xe0,0x0c,0x74]

xvsub.w $xr14, $xr23, $xr6
# CHECK-INST: xvsub.w $xr14, $xr23, $xr6
# CHECK-ENCODING: encoding: [0xee,0x1a,0x0d,0x74]

xvsub.d $xr5, $xr13, $xr7
# CHECK-INST: xvsub.d $xr5, $xr13, $xr7
# CHECK-ENCODING: encoding: [0xa5,0x9d,0x0d,0x74]

xvsub.q $xr13, $xr26, $xr31
# CHECK-INST: xvsub.q $xr13, $xr26, $xr31
# CHECK-ENCODING: encoding: [0x4d,0xff,0x2d,0x75]
