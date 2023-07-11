# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsrlni.b.h $xr5, $xr8, 2
# CHECK-INST: xvsrlni.b.h $xr5, $xr8, 2
# CHECK-ENCODING: encoding: [0x05,0x49,0x40,0x77]

xvsrlni.h.w $xr7, $xr4, 20
# CHECK-INST: xvsrlni.h.w $xr7, $xr4, 20
# CHECK-ENCODING: encoding: [0x87,0xd0,0x40,0x77]

xvsrlni.w.d $xr30, $xr15, 17
# CHECK-INST: xvsrlni.w.d $xr30, $xr15, 17
# CHECK-ENCODING: encoding: [0xfe,0x45,0x41,0x77]

xvsrlni.d.q $xr15, $xr28, 95
# CHECK-INST: xvsrlni.d.q $xr15, $xr28, 95
# CHECK-ENCODING: encoding: [0x8f,0x7f,0x43,0x77]
