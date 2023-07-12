# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvilvl.b $xr29, $xr14, $xr0
# CHECK-INST: xvilvl.b $xr29, $xr14, $xr0
# CHECK-ENCODING: encoding: [0xdd,0x01,0x1a,0x75]

xvilvl.h $xr30, $xr9, $xr21
# CHECK-INST: xvilvl.h $xr30, $xr9, $xr21
# CHECK-ENCODING: encoding: [0x3e,0xd5,0x1a,0x75]

xvilvl.w $xr24, $xr22, $xr9
# CHECK-INST: xvilvl.w $xr24, $xr22, $xr9
# CHECK-ENCODING: encoding: [0xd8,0x26,0x1b,0x75]

xvilvl.d $xr25, $xr20, $xr10
# CHECK-INST: xvilvl.d $xr25, $xr20, $xr10
# CHECK-ENCODING: encoding: [0x99,0xaa,0x1b,0x75]

xvilvh.b $xr19, $xr22, $xr26
# CHECK-INST: xvilvh.b $xr19, $xr22, $xr26
# CHECK-ENCODING: encoding: [0xd3,0x6a,0x1c,0x75]

xvilvh.h $xr10, $xr23, $xr7
# CHECK-INST: xvilvh.h $xr10, $xr23, $xr7
# CHECK-ENCODING: encoding: [0xea,0x9e,0x1c,0x75]

xvilvh.w $xr5, $xr0, $xr30
# CHECK-INST: xvilvh.w $xr5, $xr0, $xr30
# CHECK-ENCODING: encoding: [0x05,0x78,0x1d,0x75]

xvilvh.d $xr24, $xr2, $xr2
# CHECK-INST: xvilvh.d $xr24, $xr2, $xr2
# CHECK-ENCODING: encoding: [0x58,0x88,0x1d,0x75]
