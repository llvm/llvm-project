# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvdiv.b $xr9, $xr25, $xr8
# CHECK-INST: xvdiv.b $xr9, $xr25, $xr8
# CHECK-ENCODING: encoding: [0x29,0x23,0xe0,0x74]

xvdiv.h $xr18, $xr1, $xr27
# CHECK-INST: xvdiv.h $xr18, $xr1, $xr27
# CHECK-ENCODING: encoding: [0x32,0xec,0xe0,0x74]

xvdiv.w $xr5, $xr26, $xr27
# CHECK-INST: xvdiv.w $xr5, $xr26, $xr27
# CHECK-ENCODING: encoding: [0x45,0x6f,0xe1,0x74]

xvdiv.d $xr27, $xr26, $xr12
# CHECK-INST: xvdiv.d $xr27, $xr26, $xr12
# CHECK-ENCODING: encoding: [0x5b,0xb3,0xe1,0x74]

xvdiv.bu $xr0, $xr22, $xr30
# CHECK-INST: xvdiv.bu $xr0, $xr22, $xr30
# CHECK-ENCODING: encoding: [0xc0,0x7a,0xe4,0x74]

xvdiv.hu $xr31, $xr23, $xr25
# CHECK-INST: xvdiv.hu $xr31, $xr23, $xr25
# CHECK-ENCODING: encoding: [0xff,0xe6,0xe4,0x74]

xvdiv.wu $xr1, $xr25, $xr7
# CHECK-INST: xvdiv.wu $xr1, $xr25, $xr7
# CHECK-ENCODING: encoding: [0x21,0x1f,0xe5,0x74]

xvdiv.du $xr7, $xr25, $xr7
# CHECK-INST: xvdiv.du $xr7, $xr25, $xr7
# CHECK-ENCODING: encoding: [0x27,0x9f,0xe5,0x74]
