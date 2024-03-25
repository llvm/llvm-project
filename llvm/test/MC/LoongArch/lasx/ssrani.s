# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvssrani.b.h $xr26, $xr22, 14
# CHECK-INST: xvssrani.b.h $xr26, $xr22, 14
# CHECK-ENCODING: encoding: [0xda,0x7a,0x60,0x77]

xvssrani.h.w $xr19, $xr14, 26
# CHECK-INST: xvssrani.h.w $xr19, $xr14, 26
# CHECK-ENCODING: encoding: [0xd3,0xe9,0x60,0x77]

xvssrani.w.d $xr1, $xr27, 27
# CHECK-INST: xvssrani.w.d $xr1, $xr27, 27
# CHECK-ENCODING: encoding: [0x61,0x6f,0x61,0x77]

xvssrani.d.q $xr9, $xr10, 59
# CHECK-INST: xvssrani.d.q $xr9, $xr10, 59
# CHECK-ENCODING: encoding: [0x49,0xed,0x62,0x77]

xvssrani.bu.h $xr6, $xr3, 10
# CHECK-INST: xvssrani.bu.h $xr6, $xr3, 10
# CHECK-ENCODING: encoding: [0x66,0x68,0x64,0x77]

xvssrani.hu.w $xr20, $xr9, 6
# CHECK-INST: xvssrani.hu.w $xr20, $xr9, 6
# CHECK-ENCODING: encoding: [0x34,0x99,0x64,0x77]

xvssrani.wu.d $xr24, $xr11, 8
# CHECK-INST: xvssrani.wu.d $xr24, $xr11, 8
# CHECK-ENCODING: encoding: [0x78,0x21,0x65,0x77]

xvssrani.du.q $xr16, $xr2, 15
# CHECK-INST: xvssrani.du.q $xr16, $xr2, 15
# CHECK-ENCODING: encoding: [0x50,0x3c,0x66,0x77]
