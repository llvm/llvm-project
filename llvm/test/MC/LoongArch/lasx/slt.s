# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvslt.b $xr30, $xr31, $xr13
# CHECK-INST: xvslt.b $xr30, $xr31, $xr13
# CHECK-ENCODING: encoding: [0xfe,0x37,0x06,0x74]

xvslt.h $xr19, $xr23, $xr0
# CHECK-INST: xvslt.h $xr19, $xr23, $xr0
# CHECK-ENCODING: encoding: [0xf3,0x82,0x06,0x74]

xvslt.w $xr23, $xr26, $xr3
# CHECK-INST: xvslt.w $xr23, $xr26, $xr3
# CHECK-ENCODING: encoding: [0x57,0x0f,0x07,0x74]

xvslt.d $xr3, $xr10, $xr31
# CHECK-INST: xvslt.d $xr3, $xr10, $xr31
# CHECK-ENCODING: encoding: [0x43,0xfd,0x07,0x74]

xvslti.b $xr31, $xr27, 6
# CHECK-INST: xvslti.b $xr31, $xr27, 6
# CHECK-ENCODING: encoding: [0x7f,0x1b,0x86,0x76]

xvslti.h $xr5, $xr19, 6
# CHECK-INST: xvslti.h $xr5, $xr19, 6
# CHECK-ENCODING: encoding: [0x65,0x9a,0x86,0x76]

xvslti.w $xr20, $xr8, 11
# CHECK-INST: xvslti.w $xr20, $xr8, 11
# CHECK-ENCODING: encoding: [0x14,0x2d,0x87,0x76]

xvslti.d $xr13, $xr18, 2
# CHECK-INST: xvslti.d $xr13, $xr18, 2
# CHECK-ENCODING: encoding: [0x4d,0x8a,0x87,0x76]

xvslt.bu $xr20, $xr13, $xr29
# CHECK-INST: xvslt.bu $xr20, $xr13, $xr29
# CHECK-ENCODING: encoding: [0xb4,0x75,0x08,0x74]

xvslt.hu $xr12, $xr29, $xr26
# CHECK-INST: xvslt.hu $xr12, $xr29, $xr26
# CHECK-ENCODING: encoding: [0xac,0xeb,0x08,0x74]

xvslt.wu $xr26, $xr25, $xr31
# CHECK-INST: xvslt.wu $xr26, $xr25, $xr31
# CHECK-ENCODING: encoding: [0x3a,0x7f,0x09,0x74]

xvslt.du $xr30, $xr20, $xr3
# CHECK-INST: xvslt.du $xr30, $xr20, $xr3
# CHECK-ENCODING: encoding: [0x9e,0x8e,0x09,0x74]

xvslti.bu $xr1, $xr4, 2
# CHECK-INST: xvslti.bu $xr1, $xr4, 2
# CHECK-ENCODING: encoding: [0x81,0x08,0x88,0x76]

xvslti.hu $xr0, $xr5, 20
# CHECK-INST: xvslti.hu $xr0, $xr5, 20
# CHECK-ENCODING: encoding: [0xa0,0xd0,0x88,0x76]

xvslti.wu $xr0, $xr25, 24
# CHECK-INST: xvslti.wu $xr0, $xr25, 24
# CHECK-ENCODING: encoding: [0x20,0x63,0x89,0x76]

xvslti.du $xr10, $xr5, 29
# CHECK-INST: xvslti.du $xr10, $xr5, 29
# CHECK-ENCODING: encoding: [0xaa,0xf4,0x89,0x76]
