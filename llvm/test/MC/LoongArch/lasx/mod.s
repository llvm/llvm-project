# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmod.b $xr8, $xr3, $xr0
# CHECK-INST: xvmod.b $xr8, $xr3, $xr0
# CHECK-ENCODING: encoding: [0x68,0x00,0xe2,0x74]

xvmod.h $xr2, $xr17, $xr28
# CHECK-INST: xvmod.h $xr2, $xr17, $xr28
# CHECK-ENCODING: encoding: [0x22,0xf2,0xe2,0x74]

xvmod.w $xr14, $xr8, $xr13
# CHECK-INST: xvmod.w $xr14, $xr8, $xr13
# CHECK-ENCODING: encoding: [0x0e,0x35,0xe3,0x74]

xvmod.d $xr11, $xr10, $xr18
# CHECK-INST: xvmod.d $xr11, $xr10, $xr18
# CHECK-ENCODING: encoding: [0x4b,0xc9,0xe3,0x74]

xvmod.bu $xr16, $xr1, $xr26
# CHECK-INST: xvmod.bu $xr16, $xr1, $xr26
# CHECK-ENCODING: encoding: [0x30,0x68,0xe6,0x74]

xvmod.hu $xr15, $xr13, $xr0
# CHECK-INST: xvmod.hu $xr15, $xr13, $xr0
# CHECK-ENCODING: encoding: [0xaf,0x81,0xe6,0x74]

xvmod.wu $xr11, $xr19, $xr20
# CHECK-INST: xvmod.wu $xr11, $xr19, $xr20
# CHECK-ENCODING: encoding: [0x6b,0x52,0xe7,0x74]

xvmod.du $xr14, $xr3, $xr6
# CHECK-INST: xvmod.du $xr14, $xr3, $xr6
# CHECK-ENCODING: encoding: [0x6e,0x98,0xe7,0x74]
