# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsrar.b $xr9, $xr18, $xr11
# CHECK-INST: xvsrar.b $xr9, $xr18, $xr11
# CHECK-ENCODING: encoding: [0x49,0x2e,0xf2,0x74]

xvsrar.h $xr15, $xr26, $xr1
# CHECK-INST: xvsrar.h $xr15, $xr26, $xr1
# CHECK-ENCODING: encoding: [0x4f,0x87,0xf2,0x74]

xvsrar.w $xr17, $xr19, $xr14
# CHECK-INST: xvsrar.w $xr17, $xr19, $xr14
# CHECK-ENCODING: encoding: [0x71,0x3a,0xf3,0x74]

xvsrar.d $xr19, $xr15, $xr6
# CHECK-INST: xvsrar.d $xr19, $xr15, $xr6
# CHECK-ENCODING: encoding: [0xf3,0x99,0xf3,0x74]

xvsrari.b $xr10, $xr28, 3
# CHECK-INST: xvsrari.b $xr10, $xr28, 3
# CHECK-ENCODING: encoding: [0x8a,0x2f,0xa8,0x76]

xvsrari.h $xr28, $xr1, 14
# CHECK-INST: xvsrari.h $xr28, $xr1, 14
# CHECK-ENCODING: encoding: [0x3c,0x78,0xa8,0x76]

xvsrari.w $xr13, $xr7, 12
# CHECK-INST: xvsrari.w $xr13, $xr7, 12
# CHECK-ENCODING: encoding: [0xed,0xb0,0xa8,0x76]

xvsrari.d $xr29, $xr9, 8
# CHECK-INST: xvsrari.d $xr29, $xr9, 8
# CHECK-ENCODING: encoding: [0x3d,0x21,0xa9,0x76]
