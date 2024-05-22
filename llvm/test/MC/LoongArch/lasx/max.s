# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmax.b $xr23, $xr8, $xr13
# CHECK-INST: xvmax.b $xr23, $xr8, $xr13
# CHECK-ENCODING: encoding: [0x17,0x35,0x70,0x74]

xvmax.h $xr13, $xr18, $xr28
# CHECK-INST: xvmax.h $xr13, $xr18, $xr28
# CHECK-ENCODING: encoding: [0x4d,0xf2,0x70,0x74]

xvmax.w $xr26, $xr1, $xr2
# CHECK-INST: xvmax.w $xr26, $xr1, $xr2
# CHECK-ENCODING: encoding: [0x3a,0x08,0x71,0x74]

xvmax.d $xr2, $xr17, $xr13
# CHECK-INST: xvmax.d $xr2, $xr17, $xr13
# CHECK-ENCODING: encoding: [0x22,0xb6,0x71,0x74]

xvmaxi.b $xr6, $xr7, 1
# CHECK-INST: xvmaxi.b $xr6, $xr7, 1
# CHECK-ENCODING: encoding: [0xe6,0x04,0x90,0x76]

xvmaxi.h $xr24, $xr10, -7
# CHECK-INST: xvmaxi.h $xr24, $xr10, -7
# CHECK-ENCODING: encoding: [0x58,0xe5,0x90,0x76]

xvmaxi.w $xr24, $xr18, -8
# CHECK-INST: xvmaxi.w $xr24, $xr18, -8
# CHECK-ENCODING: encoding: [0x58,0x62,0x91,0x76]

xvmaxi.d $xr21, $xr5, -11
# CHECK-INST: xvmaxi.d $xr21, $xr5, -11
# CHECK-ENCODING: encoding: [0xb5,0xd4,0x91,0x76]

xvmax.bu $xr29, $xr30, $xr11
# CHECK-INST: xvmax.bu $xr29, $xr30, $xr11
# CHECK-ENCODING: encoding: [0xdd,0x2f,0x74,0x74]

xvmax.hu $xr4, $xr23, $xr27
# CHECK-INST: xvmax.hu $xr4, $xr23, $xr27
# CHECK-ENCODING: encoding: [0xe4,0xee,0x74,0x74]

xvmax.wu $xr31, $xr0, $xr0
# CHECK-INST: xvmax.wu $xr31, $xr0, $xr0
# CHECK-ENCODING: encoding: [0x1f,0x00,0x75,0x74]

xvmax.du $xr5, $xr22, $xr9
# CHECK-INST: xvmax.du $xr5, $xr22, $xr9
# CHECK-ENCODING: encoding: [0xc5,0xa6,0x75,0x74]

xvmaxi.bu $xr12, $xr27, 28
# CHECK-INST: xvmaxi.bu $xr12, $xr27, 28
# CHECK-ENCODING: encoding: [0x6c,0x73,0x94,0x76]

xvmaxi.hu $xr25, $xr4, 16
# CHECK-INST: xvmaxi.hu $xr25, $xr4, 16
# CHECK-ENCODING: encoding: [0x99,0xc0,0x94,0x76]

xvmaxi.wu $xr27, $xr7, 21
# CHECK-INST: xvmaxi.wu $xr27, $xr7, 21
# CHECK-ENCODING: encoding: [0xfb,0x54,0x95,0x76]

xvmaxi.du $xr31, $xr13, 9
# CHECK-INST: xvmaxi.du $xr31, $xr13, 9
# CHECK-ENCODING: encoding: [0xbf,0xa5,0x95,0x76]
