# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsrlrni.b.h $xr10, $xr17, 12
# CHECK-INST: xvsrlrni.b.h $xr10, $xr17, 12
# CHECK-ENCODING: encoding: [0x2a,0x72,0x44,0x77]

xvsrlrni.h.w $xr22, $xr23, 13
# CHECK-INST: xvsrlrni.h.w $xr22, $xr23, 13
# CHECK-ENCODING: encoding: [0xf6,0xb6,0x44,0x77]

xvsrlrni.w.d $xr18, $xr22, 58
# CHECK-INST: xvsrlrni.w.d $xr18, $xr22, 58
# CHECK-ENCODING: encoding: [0xd2,0xea,0x45,0x77]

xvsrlrni.d.q $xr25, $xr8, 42
# CHECK-INST: xvsrlrni.d.q $xr25, $xr8, 42
# CHECK-ENCODING: encoding: [0x19,0xa9,0x46,0x77]
