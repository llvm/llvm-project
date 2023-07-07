# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvexth.h.b $xr15, $xr10
# CHECK-INST: xvexth.h.b $xr15, $xr10
# CHECK-ENCODING: encoding: [0x4f,0xe1,0x9e,0x76]

xvexth.w.h $xr26, $xr11
# CHECK-INST: xvexth.w.h $xr26, $xr11
# CHECK-ENCODING: encoding: [0x7a,0xe5,0x9e,0x76]

xvexth.d.w $xr2, $xr27
# CHECK-INST: xvexth.d.w $xr2, $xr27
# CHECK-ENCODING: encoding: [0x62,0xeb,0x9e,0x76]

xvexth.q.d $xr22, $xr25
# CHECK-INST: xvexth.q.d $xr22, $xr25
# CHECK-ENCODING: encoding: [0x36,0xef,0x9e,0x76]

xvexth.hu.bu $xr21, $xr30
# CHECK-INST: xvexth.hu.bu $xr21, $xr30
# CHECK-ENCODING: encoding: [0xd5,0xf3,0x9e,0x76]

xvexth.wu.hu $xr28, $xr11
# CHECK-INST: xvexth.wu.hu $xr28, $xr11
# CHECK-ENCODING: encoding: [0x7c,0xf5,0x9e,0x76]

xvexth.du.wu $xr27, $xr25
# CHECK-INST: xvexth.du.wu $xr27, $xr25
# CHECK-ENCODING: encoding: [0x3b,0xfb,0x9e,0x76]

xvexth.qu.du $xr16, $xr28
# CHECK-INST: xvexth.qu.du $xr16, $xr28
# CHECK-ENCODING: encoding: [0x90,0xff,0x9e,0x76]
