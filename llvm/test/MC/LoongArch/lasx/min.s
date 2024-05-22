# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvmin.b $xr21, $xr26, $xr7
# CHECK-INST: xvmin.b $xr21, $xr26, $xr7
# CHECK-ENCODING: encoding: [0x55,0x1f,0x72,0x74]

xvmin.h $xr29, $xr5, $xr9
# CHECK-INST: xvmin.h $xr29, $xr5, $xr9
# CHECK-ENCODING: encoding: [0xbd,0xa4,0x72,0x74]

xvmin.w $xr31, $xr24, $xr20
# CHECK-INST: xvmin.w $xr31, $xr24, $xr20
# CHECK-ENCODING: encoding: [0x1f,0x53,0x73,0x74]

xvmin.d $xr27, $xr27, $xr2
# CHECK-INST: xvmin.d $xr27, $xr27, $xr2
# CHECK-ENCODING: encoding: [0x7b,0x8b,0x73,0x74]

xvmini.b $xr22, $xr17, 9
# CHECK-INST: xvmini.b $xr22, $xr17, 9
# CHECK-ENCODING: encoding: [0x36,0x26,0x92,0x76]

xvmini.h $xr12, $xr23, -15
# CHECK-INST: xvmini.h $xr12, $xr23, -15
# CHECK-ENCODING: encoding: [0xec,0xc6,0x92,0x76]

xvmini.w $xr1, $xr17, -13
# CHECK-INST: xvmini.w $xr1, $xr17, -13
# CHECK-ENCODING: encoding: [0x21,0x4e,0x93,0x76]

xvmini.d $xr10, $xr31, 11
# CHECK-INST: xvmini.d $xr10, $xr31, 11
# CHECK-ENCODING: encoding: [0xea,0xaf,0x93,0x76]

xvmin.bu $xr15, $xr16, $xr3
# CHECK-INST: xvmin.bu $xr15, $xr16, $xr3
# CHECK-ENCODING: encoding: [0x0f,0x0e,0x76,0x74]

xvmin.hu $xr4, $xr31, $xr27
# CHECK-INST: xvmin.hu $xr4, $xr31, $xr27
# CHECK-ENCODING: encoding: [0xe4,0xef,0x76,0x74]

xvmin.wu $xr15, $xr13, $xr28
# CHECK-INST: xvmin.wu $xr15, $xr13, $xr28
# CHECK-ENCODING: encoding: [0xaf,0x71,0x77,0x74]

xvmin.du $xr27, $xr3, $xr5
# CHECK-INST: xvmin.du $xr27, $xr3, $xr5
# CHECK-ENCODING: encoding: [0x7b,0x94,0x77,0x74]

xvmini.bu $xr6, $xr24, 7
# CHECK-INST: xvmini.bu $xr6, $xr24, 7
# CHECK-ENCODING: encoding: [0x06,0x1f,0x96,0x76]

xvmini.hu $xr8, $xr5, 29
# CHECK-INST: xvmini.hu $xr8, $xr5, 29
# CHECK-ENCODING: encoding: [0xa8,0xf4,0x96,0x76]

xvmini.wu $xr17, $xr13, 19
# CHECK-INST: xvmini.wu $xr17, $xr13, 19
# CHECK-ENCODING: encoding: [0xb1,0x4d,0x97,0x76]

xvmini.du $xr16, $xr23, 30
# CHECK-INST: xvmini.du $xr16, $xr23, 30
# CHECK-ENCODING: encoding: [0xf0,0xfa,0x97,0x76]
