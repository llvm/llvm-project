# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsrarni.b.h $xr21, $xr31, 15
# CHECK-INST: xvsrarni.b.h $xr21, $xr31, 15
# CHECK-ENCODING: encoding: [0xf5,0x7f,0x5c,0x77]

xvsrarni.h.w $xr4, $xr22, 25
# CHECK-INST: xvsrarni.h.w $xr4, $xr22, 25
# CHECK-ENCODING: encoding: [0xc4,0xe6,0x5c,0x77]

xvsrarni.w.d $xr24, $xr8, 41
# CHECK-INST: xvsrarni.w.d $xr24, $xr8, 41
# CHECK-ENCODING: encoding: [0x18,0xa5,0x5d,0x77]

xvsrarni.d.q $xr7, $xr5, 7
# CHECK-INST: xvsrarni.d.q $xr7, $xr5, 7
# CHECK-ENCODING: encoding: [0xa7,0x1c,0x5e,0x77]
