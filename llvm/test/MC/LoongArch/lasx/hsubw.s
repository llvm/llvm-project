# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvhsubw.h.b $xr22, $xr7, $xr16
# CHECK-INST: xvhsubw.h.b $xr22, $xr7, $xr16
# CHECK-ENCODING: encoding: [0xf6,0x40,0x56,0x74]

xvhsubw.w.h $xr19, $xr8, $xr15
# CHECK-INST: xvhsubw.w.h $xr19, $xr8, $xr15
# CHECK-ENCODING: encoding: [0x13,0xbd,0x56,0x74]

xvhsubw.d.w $xr30, $xr23, $xr19
# CHECK-INST: xvhsubw.d.w $xr30, $xr23, $xr19
# CHECK-ENCODING: encoding: [0xfe,0x4e,0x57,0x74]

xvhsubw.q.d $xr20, $xr13, $xr28
# CHECK-INST: xvhsubw.q.d $xr20, $xr13, $xr28
# CHECK-ENCODING: encoding: [0xb4,0xf1,0x57,0x74]

xvhsubw.hu.bu $xr10, $xr2, $xr16
# CHECK-INST: xvhsubw.hu.bu $xr10, $xr2, $xr16
# CHECK-ENCODING: encoding: [0x4a,0x40,0x5a,0x74]

xvhsubw.wu.hu $xr1, $xr26, $xr18
# CHECK-INST: xvhsubw.wu.hu $xr1, $xr26, $xr18
# CHECK-ENCODING: encoding: [0x41,0xcb,0x5a,0x74]

xvhsubw.du.wu $xr5, $xr23, $xr20
# CHECK-INST: xvhsubw.du.wu $xr5, $xr23, $xr20
# CHECK-ENCODING: encoding: [0xe5,0x52,0x5b,0x74]

xvhsubw.qu.du $xr31, $xr4, $xr8
# CHECK-INST: xvhsubw.qu.du $xr31, $xr4, $xr8
# CHECK-ENCODING: encoding: [0x9f,0xa0,0x5b,0x74]
