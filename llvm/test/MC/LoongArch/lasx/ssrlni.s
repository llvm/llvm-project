# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvssrlni.b.h $xr19, $xr18, 9
# CHECK-INST: xvssrlni.b.h $xr19, $xr18, 9
# CHECK-ENCODING: encoding: [0x53,0x66,0x48,0x77]

xvssrlni.h.w $xr29, $xr29, 3
# CHECK-INST: xvssrlni.h.w $xr29, $xr29, 3
# CHECK-ENCODING: encoding: [0xbd,0x8f,0x48,0x77]

xvssrlni.w.d $xr9, $xr15, 43
# CHECK-INST: xvssrlni.w.d $xr9, $xr15, 43
# CHECK-ENCODING: encoding: [0xe9,0xad,0x49,0x77]

xvssrlni.d.q $xr8, $xr11, 121
# CHECK-INST: xvssrlni.d.q $xr8, $xr11, 121
# CHECK-ENCODING: encoding: [0x68,0xe5,0x4b,0x77]

xvssrlni.bu.h $xr25, $xr10, 5
# CHECK-INST: xvssrlni.bu.h $xr25, $xr10, 5
# CHECK-ENCODING: encoding: [0x59,0x55,0x4c,0x77]

xvssrlni.hu.w $xr9, $xr18, 26
# CHECK-INST: xvssrlni.hu.w $xr9, $xr18, 26
# CHECK-ENCODING: encoding: [0x49,0xea,0x4c,0x77]

xvssrlni.wu.d $xr20, $xr22, 13
# CHECK-INST: xvssrlni.wu.d $xr20, $xr22, 13
# CHECK-ENCODING: encoding: [0xd4,0x36,0x4d,0x77]

xvssrlni.du.q $xr8, $xr4, 43
# CHECK-INST: xvssrlni.du.q $xr8, $xr4, 43
# CHECK-ENCODING: encoding: [0x88,0xac,0x4e,0x77]
