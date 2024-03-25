# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsrani.b.h $xr14, $xr23, 15
# CHECK-INST: xvsrani.b.h $xr14, $xr23, 15
# CHECK-ENCODING: encoding: [0xee,0x7e,0x58,0x77]

xvsrani.h.w $xr2, $xr8, 5
# CHECK-INST: xvsrani.h.w $xr2, $xr8, 5
# CHECK-ENCODING: encoding: [0x02,0x95,0x58,0x77]

xvsrani.w.d $xr5, $xr11, 14
# CHECK-INST: xvsrani.w.d $xr5, $xr11, 14
# CHECK-ENCODING: encoding: [0x65,0x39,0x59,0x77]

xvsrani.d.q $xr17, $xr7, 113
# CHECK-INST: xvsrani.d.q $xr17, $xr7, 113
# CHECK-ENCODING: encoding: [0xf1,0xc4,0x5b,0x77]
