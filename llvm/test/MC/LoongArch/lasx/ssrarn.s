# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvssrarn.b.h $xr7, $xr13, $xr0
# CHECK-INST: xvssrarn.b.h $xr7, $xr13, $xr0
# CHECK-ENCODING: encoding: [0xa7,0x81,0x02,0x75]

xvssrarn.h.w $xr22, $xr2, $xr14
# CHECK-INST: xvssrarn.h.w $xr22, $xr2, $xr14
# CHECK-ENCODING: encoding: [0x56,0x38,0x03,0x75]

xvssrarn.w.d $xr13, $xr7, $xr16
# CHECK-INST: xvssrarn.w.d $xr13, $xr7, $xr16
# CHECK-ENCODING: encoding: [0xed,0xc0,0x03,0x75]

xvssrarn.bu.h $xr4, $xr12, $xr2
# CHECK-INST: xvssrarn.bu.h $xr4, $xr12, $xr2
# CHECK-ENCODING: encoding: [0x84,0x89,0x0a,0x75]

xvssrarn.hu.w $xr15, $xr24, $xr3
# CHECK-INST: xvssrarn.hu.w $xr15, $xr24, $xr3
# CHECK-ENCODING: encoding: [0x0f,0x0f,0x0b,0x75]

xvssrarn.wu.d $xr30, $xr9, $xr8
# CHECK-INST: xvssrarn.wu.d $xr30, $xr9, $xr8
# CHECK-ENCODING: encoding: [0x3e,0xa1,0x0b,0x75]
