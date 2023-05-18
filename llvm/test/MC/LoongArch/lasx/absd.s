# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvabsd.b $xr22, $xr1, $xr17
# CHECK-INST: xvabsd.b $xr22, $xr1, $xr17
# CHECK-ENCODING: encoding: [0x36,0x44,0x60,0x74]

xvabsd.h $xr17, $xr24, $xr9
# CHECK-INST: xvabsd.h $xr17, $xr24, $xr9
# CHECK-ENCODING: encoding: [0x11,0xa7,0x60,0x74]

xvabsd.w $xr28, $xr9, $xr29
# CHECK-INST: xvabsd.w $xr28, $xr9, $xr29
# CHECK-ENCODING: encoding: [0x3c,0x75,0x61,0x74]

xvabsd.d $xr30, $xr23, $xr19
# CHECK-INST: xvabsd.d $xr30, $xr23, $xr19
# CHECK-ENCODING: encoding: [0xfe,0xce,0x61,0x74]

xvabsd.bu $xr16, $xr4, $xr15
# CHECK-INST: xvabsd.bu $xr16, $xr4, $xr15
# CHECK-ENCODING: encoding: [0x90,0x3c,0x62,0x74]

xvabsd.hu $xr13, $xr23, $xr27
# CHECK-INST: xvabsd.hu $xr13, $xr23, $xr27
# CHECK-ENCODING: encoding: [0xed,0xee,0x62,0x74]

xvabsd.wu $xr31, $xr18, $xr15
# CHECK-INST: xvabsd.wu $xr31, $xr18, $xr15
# CHECK-ENCODING: encoding: [0x5f,0x3e,0x63,0x74]

xvabsd.du $xr26, $xr10, $xr4
# CHECK-INST: xvabsd.du $xr26, $xr10, $xr4
# CHECK-ENCODING: encoding: [0x5a,0x91,0x63,0x74]
