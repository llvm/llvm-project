# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvssrarni.b.h $xr0, $xr4, 13
# CHECK-INST: xvssrarni.b.h $xr0, $xr4, 13
# CHECK-ENCODING: encoding: [0x80,0x74,0x68,0x77]

xvssrarni.h.w $xr8, $xr0, 9
# CHECK-INST: xvssrarni.h.w $xr8, $xr0, 9
# CHECK-ENCODING: encoding: [0x08,0xa4,0x68,0x77]

xvssrarni.w.d $xr5, $xr5, 42
# CHECK-INST: xvssrarni.w.d $xr5, $xr5, 42
# CHECK-ENCODING: encoding: [0xa5,0xa8,0x69,0x77]

xvssrarni.d.q $xr8, $xr31, 83
# CHECK-INST: xvssrarni.d.q $xr8, $xr31, 83
# CHECK-ENCODING: encoding: [0xe8,0x4f,0x6b,0x77]

xvssrarni.bu.h $xr21, $xr19, 0
# CHECK-INST: xvssrarni.bu.h $xr21, $xr19, 0
# CHECK-ENCODING: encoding: [0x75,0x42,0x6c,0x77]

xvssrarni.hu.w $xr22, $xr13, 1
# CHECK-INST: xvssrarni.hu.w $xr22, $xr13, 1
# CHECK-ENCODING: encoding: [0xb6,0x85,0x6c,0x77]

xvssrarni.wu.d $xr21, $xr5, 26
# CHECK-INST: xvssrarni.wu.d $xr21, $xr5, 26
# CHECK-ENCODING: encoding: [0xb5,0x68,0x6d,0x77]

xvssrarni.du.q $xr15, $xr14, 94
# CHECK-INST: xvssrarni.du.q $xr15, $xr14, 94
# CHECK-ENCODING: encoding: [0xcf,0x79,0x6f,0x77]
