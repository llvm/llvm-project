# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvbitclr.b $xr24, $xr5, $xr14
# CHECK-INST: xvbitclr.b $xr24, $xr5, $xr14
# CHECK-ENCODING: encoding: [0xb8,0x38,0x0c,0x75]

xvbitclr.h $xr30, $xr9, $xr13
# CHECK-INST: xvbitclr.h $xr30, $xr9, $xr13
# CHECK-ENCODING: encoding: [0x3e,0xb5,0x0c,0x75]

xvbitclr.w $xr2, $xr3, $xr7
# CHECK-INST: xvbitclr.w $xr2, $xr3, $xr7
# CHECK-ENCODING: encoding: [0x62,0x1c,0x0d,0x75]

xvbitclr.d $xr14, $xr5, $xr25
# CHECK-INST: xvbitclr.d $xr14, $xr5, $xr25
# CHECK-ENCODING: encoding: [0xae,0xe4,0x0d,0x75]

xvbitclri.b $xr22, $xr26, 7
# CHECK-INST: xvbitclri.b $xr22, $xr26, 7
# CHECK-ENCODING: encoding: [0x56,0x3f,0x10,0x77]

xvbitclri.h $xr2, $xr14, 13
# CHECK-INST: xvbitclri.h $xr2, $xr14, 13
# CHECK-ENCODING: encoding: [0xc2,0x75,0x10,0x77]

xvbitclri.w $xr3, $xr2, 0
# CHECK-INST: xvbitclri.w $xr3, $xr2, 0
# CHECK-ENCODING: encoding: [0x43,0x80,0x10,0x77]

xvbitclri.d $xr10, $xr12, 7
# CHECK-INST: xvbitclri.d $xr10, $xr12, 7
# CHECK-ENCODING: encoding: [0x8a,0x1d,0x11,0x77]
