# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsetallnez.b $fcc5, $xr29
# CHECK-INST: xvsetallnez.b $fcc5, $xr29
# CHECK-ENCODING: encoding: [0xa5,0xb3,0x9c,0x76]

xvsetallnez.h $fcc5, $xr4
# CHECK-INST: xvsetallnez.h $fcc5, $xr4
# CHECK-ENCODING: encoding: [0x85,0xb4,0x9c,0x76]

xvsetallnez.w $fcc4, $xr5
# CHECK-INST: xvsetallnez.w $fcc4, $xr5
# CHECK-ENCODING: encoding: [0xa4,0xb8,0x9c,0x76]

xvsetallnez.d $fcc7, $xr20
# CHECK-INST: xvsetallnez.d $fcc7, $xr20
# CHECK-ENCODING: encoding: [0x87,0xbe,0x9c,0x76]
