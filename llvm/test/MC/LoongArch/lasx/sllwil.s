# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsllwil.h.b $xr13, $xr21, 6
# CHECK-INST: xvsllwil.h.b $xr13, $xr21, 6
# CHECK-ENCODING: encoding: [0xad,0x3a,0x08,0x77]

xvsllwil.w.h $xr20, $xr29, 0
# CHECK-INST: xvsllwil.w.h $xr20, $xr29, 0
# CHECK-ENCODING: encoding: [0xb4,0x43,0x08,0x77]

xvsllwil.d.w $xr3, $xr20, 24
# CHECK-INST: xvsllwil.d.w $xr3, $xr20, 24
# CHECK-ENCODING: encoding: [0x83,0xe2,0x08,0x77]

xvsllwil.hu.bu $xr15, $xr15, 6
# CHECK-INST: xvsllwil.hu.bu $xr15, $xr15, 6
# CHECK-ENCODING: encoding: [0xef,0x39,0x0c,0x77]

xvsllwil.wu.hu $xr22, $xr29, 0
# CHECK-INST: xvsllwil.wu.hu $xr22, $xr29, 0
# CHECK-ENCODING: encoding: [0xb6,0x43,0x0c,0x77]

xvsllwil.du.wu $xr3, $xr5, 31
# CHECK-INST: xvsllwil.du.wu $xr3, $xr5, 31
# CHECK-ENCODING: encoding: [0xa3,0xfc,0x0c,0x77]
