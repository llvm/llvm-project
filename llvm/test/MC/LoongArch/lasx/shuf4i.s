# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvshuf4i.b $xr21, $xr28, 168
# CHECK-INST: xvshuf4i.b $xr21, $xr28, 168
# CHECK-ENCODING: encoding: [0x95,0xa3,0x92,0x77]

xvshuf4i.h $xr18, $xr3, 22
# CHECK-INST: xvshuf4i.h $xr18, $xr3, 22
# CHECK-ENCODING: encoding: [0x72,0x58,0x94,0x77]

xvshuf4i.w $xr0, $xr25, 82
# CHECK-INST: xvshuf4i.w $xr0, $xr25, 82
# CHECK-ENCODING: encoding: [0x20,0x4b,0x99,0x77]

xvshuf4i.d $xr24, $xr4, 99
# CHECK-INST: xvshuf4i.d $xr24, $xr4, 99
# CHECK-ENCODING: encoding: [0x98,0x8c,0x9d,0x77]
