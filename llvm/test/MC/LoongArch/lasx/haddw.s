# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvhaddw.h.b $xr31, $xr19, $xr29
# CHECK-INST: xvhaddw.h.b $xr31, $xr19, $xr29
# CHECK-ENCODING: encoding: [0x7f,0x76,0x54,0x74]

xvhaddw.w.h $xr31, $xr16, $xr23
# CHECK-INST: xvhaddw.w.h $xr31, $xr16, $xr23
# CHECK-ENCODING: encoding: [0x1f,0xde,0x54,0x74]

xvhaddw.d.w $xr30, $xr1, $xr24
# CHECK-INST: xvhaddw.d.w $xr30, $xr1, $xr24
# CHECK-ENCODING: encoding: [0x3e,0x60,0x55,0x74]

xvhaddw.q.d $xr16, $xr15, $xr17
# CHECK-INST: xvhaddw.q.d $xr16, $xr15, $xr17
# CHECK-ENCODING: encoding: [0xf0,0xc5,0x55,0x74]

xvhaddw.hu.bu $xr14, $xr17, $xr2
# CHECK-INST: xvhaddw.hu.bu $xr14, $xr17, $xr2
# CHECK-ENCODING: encoding: [0x2e,0x0a,0x58,0x74]

xvhaddw.wu.hu $xr21, $xr2, $xr8
# CHECK-INST: xvhaddw.wu.hu $xr21, $xr2, $xr8
# CHECK-ENCODING: encoding: [0x55,0xa0,0x58,0x74]

xvhaddw.du.wu $xr6, $xr24, $xr19
# CHECK-INST: xvhaddw.du.wu $xr6, $xr24, $xr19
# CHECK-ENCODING: encoding: [0x06,0x4f,0x59,0x74]

xvhaddw.qu.du $xr10, $xr12, $xr13
# CHECK-INST: xvhaddw.qu.du $xr10, $xr12, $xr13
# CHECK-ENCODING: encoding: [0x8a,0xb5,0x59,0x74]
