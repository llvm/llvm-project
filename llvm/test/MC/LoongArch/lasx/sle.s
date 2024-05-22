# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsle.b $xr24, $xr30, $xr29
# CHECK-INST: xvsle.b $xr24, $xr30, $xr29
# CHECK-ENCODING: encoding: [0xd8,0x77,0x02,0x74]

xvsle.h $xr23, $xr13, $xr20
# CHECK-INST: xvsle.h $xr23, $xr13, $xr20
# CHECK-ENCODING: encoding: [0xb7,0xd1,0x02,0x74]

xvsle.w $xr10, $xr31, $xr24
# CHECK-INST: xvsle.w $xr10, $xr31, $xr24
# CHECK-ENCODING: encoding: [0xea,0x63,0x03,0x74]

xvsle.d $xr13, $xr26, $xr8
# CHECK-INST: xvsle.d $xr13, $xr26, $xr8
# CHECK-ENCODING: encoding: [0x4d,0xa3,0x03,0x74]

xvslei.b $xr14, $xr11, -10
# CHECK-INST: xvslei.b $xr14, $xr11, -10
# CHECK-ENCODING: encoding: [0x6e,0x59,0x82,0x76]

xvslei.h $xr2, $xr22, 15
# CHECK-INST: xvslei.h $xr2, $xr22, 15
# CHECK-ENCODING: encoding: [0xc2,0xbe,0x82,0x76]

xvslei.w $xr3, $xr14, 12
# CHECK-INST: xvslei.w $xr3, $xr14, 12
# CHECK-ENCODING: encoding: [0xc3,0x31,0x83,0x76]

xvslei.d $xr19, $xr30, 10
# CHECK-INST: xvslei.d $xr19, $xr30, 10
# CHECK-ENCODING: encoding: [0xd3,0xab,0x83,0x76]

xvsle.bu $xr9, $xr27, $xr2
# CHECK-INST: xvsle.bu $xr9, $xr27, $xr2
# CHECK-ENCODING: encoding: [0x69,0x0b,0x04,0x74]

xvsle.hu $xr29, $xr25, $xr22
# CHECK-INST: xvsle.hu $xr29, $xr25, $xr22
# CHECK-ENCODING: encoding: [0x3d,0xdb,0x04,0x74]

xvsle.wu $xr16, $xr25, $xr14
# CHECK-INST: xvsle.wu $xr16, $xr25, $xr14
# CHECK-ENCODING: encoding: [0x30,0x3b,0x05,0x74]

xvsle.du $xr5, $xr6, $xr18
# CHECK-INST: xvsle.du $xr5, $xr6, $xr18
# CHECK-ENCODING: encoding: [0xc5,0xc8,0x05,0x74]

xvslei.bu $xr17, $xr26, 10
# CHECK-INST: xvslei.bu $xr17, $xr26, 10
# CHECK-ENCODING: encoding: [0x51,0x2b,0x84,0x76]

xvslei.hu $xr20, $xr11, 18
# CHECK-INST: xvslei.hu $xr20, $xr11, 18
# CHECK-ENCODING: encoding: [0x74,0xc9,0x84,0x76]

xvslei.wu $xr1, $xr29, 10
# CHECK-INST: xvslei.wu $xr1, $xr29, 10
# CHECK-ENCODING: encoding: [0xa1,0x2b,0x85,0x76]

xvslei.du $xr25, $xr31, 24
# CHECK-INST: xvslei.du $xr25, $xr31, 24
# CHECK-ENCODING: encoding: [0xf9,0xe3,0x85,0x76]
