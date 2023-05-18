# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvssrlrni.b.h $xr26, $xr26, 8
# CHECK-INST: xvssrlrni.b.h $xr26, $xr26, 8
# CHECK-ENCODING: encoding: [0x5a,0x63,0x50,0x77]

xvssrlrni.h.w $xr6, $xr0, 19
# CHECK-INST: xvssrlrni.h.w $xr6, $xr0, 19
# CHECK-ENCODING: encoding: [0x06,0xcc,0x50,0x77]

xvssrlrni.w.d $xr28, $xr15, 55
# CHECK-INST: xvssrlrni.w.d $xr28, $xr15, 55
# CHECK-ENCODING: encoding: [0xfc,0xdd,0x51,0x77]

xvssrlrni.d.q $xr8, $xr16, 64
# CHECK-INST: xvssrlrni.d.q $xr8, $xr16, 64
# CHECK-ENCODING: encoding: [0x08,0x02,0x53,0x77]

xvssrlrni.bu.h $xr23, $xr28, 3
# CHECK-INST: xvssrlrni.bu.h $xr23, $xr28, 3
# CHECK-ENCODING: encoding: [0x97,0x4f,0x54,0x77]

xvssrlrni.hu.w $xr25, $xr10, 18
# CHECK-INST: xvssrlrni.hu.w $xr25, $xr10, 18
# CHECK-ENCODING: encoding: [0x59,0xc9,0x54,0x77]

xvssrlrni.wu.d $xr16, $xr28, 15
# CHECK-INST: xvssrlrni.wu.d $xr16, $xr28, 15
# CHECK-ENCODING: encoding: [0x90,0x3f,0x55,0x77]

xvssrlrni.du.q $xr18, $xr9, 44
# CHECK-INST: xvssrlrni.du.q $xr18, $xr9, 44
# CHECK-ENCODING: encoding: [0x32,0xb1,0x56,0x77]
