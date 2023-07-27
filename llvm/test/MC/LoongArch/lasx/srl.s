# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsrl.b $xr20, $xr24, $xr29
# CHECK-INST: xvsrl.b $xr20, $xr24, $xr29
# CHECK-ENCODING: encoding: [0x14,0x77,0xea,0x74]

xvsrl.h $xr11, $xr17, $xr31
# CHECK-INST: xvsrl.h $xr11, $xr17, $xr31
# CHECK-ENCODING: encoding: [0x2b,0xfe,0xea,0x74]

xvsrl.w $xr2, $xr10, $xr8
# CHECK-INST: xvsrl.w $xr2, $xr10, $xr8
# CHECK-ENCODING: encoding: [0x42,0x21,0xeb,0x74]

xvsrl.d $xr13, $xr30, $xr26
# CHECK-INST: xvsrl.d $xr13, $xr30, $xr26
# CHECK-ENCODING: encoding: [0xcd,0xeb,0xeb,0x74]

xvsrli.b $xr29, $xr4, 3
# CHECK-INST: xvsrli.b $xr29, $xr4, 3
# CHECK-ENCODING: encoding: [0x9d,0x2c,0x30,0x77]

xvsrli.h $xr28, $xr14, 12
# CHECK-INST: xvsrli.h $xr28, $xr14, 12
# CHECK-ENCODING: encoding: [0xdc,0x71,0x30,0x77]

xvsrli.w $xr12, $xr18, 7
# CHECK-INST: xvsrli.w $xr12, $xr18, 7
# CHECK-ENCODING: encoding: [0x4c,0x9e,0x30,0x77]

xvsrli.d $xr0, $xr4, 46
# CHECK-INST: xvsrli.d $xr0, $xr4, 46
# CHECK-ENCODING: encoding: [0x80,0xb8,0x31,0x77]
