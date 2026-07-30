# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsll.b $xr8, $xr29, $xr9
# CHECK-INST: xvsll.b $xr8, $xr29, $xr9
# CHECK-ENCODING: encoding: [0xa8,0x27,0xe8,0x74]

xvsll.h $xr21, $xr28, $xr29
# CHECK-INST: xvsll.h $xr21, $xr28, $xr29
# CHECK-ENCODING: encoding: [0x95,0xf7,0xe8,0x74]

xvsll.w $xr17, $xr30, $xr10
# CHECK-INST: xvsll.w $xr17, $xr30, $xr10
# CHECK-ENCODING: encoding: [0xd1,0x2b,0xe9,0x74]

xvsll.d $xr19, $xr6, $xr26
# CHECK-INST: xvsll.d $xr19, $xr6, $xr26
# CHECK-ENCODING: encoding: [0xd3,0xe8,0xe9,0x74]

xvslli.b $xr25, $xr26, 1
# CHECK-INST: xvslli.b $xr25, $xr26, 1
# CHECK-ENCODING: encoding: [0x59,0x27,0x2c,0x77]

xvslli.h $xr17, $xr28, 14
# CHECK-INST: xvslli.h $xr17, $xr28, 14
# CHECK-ENCODING: encoding: [0x91,0x7b,0x2c,0x77]

xvslli.w $xr26, $xr31, 29
# CHECK-INST: xvslli.w $xr26, $xr31, 29
# CHECK-ENCODING: encoding: [0xfa,0xf7,0x2c,0x77]

xvslli.d $xr10, $xr28, 46
# CHECK-INST: xvslli.d $xr10, $xr28, 46
# CHECK-ENCODING: encoding: [0x8a,0xbb,0x2d,0x77]
