# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvffint.s.w $xr3, $xr5
# CHECK-INST: xvffint.s.w $xr3, $xr5
# CHECK-ENCODING: encoding: [0xa3,0x00,0x9e,0x76]

xvffint.d.l $xr5, $xr19
# CHECK-INST: xvffint.d.l $xr5, $xr19
# CHECK-ENCODING: encoding: [0x65,0x0a,0x9e,0x76]

xvffint.s.wu $xr3, $xr28
# CHECK-INST: xvffint.s.wu $xr3, $xr28
# CHECK-ENCODING: encoding: [0x83,0x07,0x9e,0x76]

xvffint.d.lu $xr31, $xr29
# CHECK-INST: xvffint.d.lu $xr31, $xr29
# CHECK-ENCODING: encoding: [0xbf,0x0f,0x9e,0x76]

xvffintl.d.w $xr2, $xr7
# CHECK-INST: xvffintl.d.w $xr2, $xr7
# CHECK-ENCODING: encoding: [0xe2,0x10,0x9e,0x76]

xvffinth.d.w $xr7, $xr28
# CHECK-INST: xvffinth.d.w $xr7, $xr28
# CHECK-ENCODING: encoding: [0x87,0x17,0x9e,0x76]

xvffint.s.l $xr10, $xr27, $xr3
# CHECK-INST: xvffint.s.l $xr10, $xr27, $xr3
# CHECK-ENCODING: encoding: [0x6a,0x0f,0x48,0x75]
