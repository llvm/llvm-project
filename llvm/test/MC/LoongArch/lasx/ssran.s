# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvssran.b.h $xr17, $xr4, $xr1
# CHECK-INST: xvssran.b.h $xr17, $xr4, $xr1
# CHECK-ENCODING: encoding: [0x91,0x84,0xfe,0x74]

xvssran.h.w $xr28, $xr28, $xr13
# CHECK-INST: xvssran.h.w $xr28, $xr28, $xr13
# CHECK-ENCODING: encoding: [0x9c,0x37,0xff,0x74]

xvssran.w.d $xr21, $xr1, $xr31
# CHECK-INST: xvssran.w.d $xr21, $xr1, $xr31
# CHECK-ENCODING: encoding: [0x35,0xfc,0xff,0x74]

xvssran.bu.h $xr3, $xr12, $xr24
# CHECK-INST: xvssran.bu.h $xr3, $xr12, $xr24
# CHECK-ENCODING: encoding: [0x83,0xe1,0x06,0x75]

xvssran.hu.w $xr25, $xr24, $xr1
# CHECK-INST: xvssran.hu.w $xr25, $xr24, $xr1
# CHECK-ENCODING: encoding: [0x19,0x07,0x07,0x75]

xvssran.wu.d $xr30, $xr14, $xr10
# CHECK-INST: xvssran.wu.d $xr30, $xr14, $xr10
# CHECK-ENCODING: encoding: [0xde,0xa9,0x07,0x75]
