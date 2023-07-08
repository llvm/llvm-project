# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfrintrne.s $xr19, $xr17
# CHECK-INST: xvfrintrne.s $xr19, $xr17
# CHECK-ENCODING: encoding: [0x33,0x76,0x9d,0x76]

xvfrintrne.d $xr12, $xr29
# CHECK-INST: xvfrintrne.d $xr12, $xr29
# CHECK-ENCODING: encoding: [0xac,0x7b,0x9d,0x76]

xvfrintrz.s $xr10, $xr9
# CHECK-INST: xvfrintrz.s $xr10, $xr9
# CHECK-ENCODING: encoding: [0x2a,0x65,0x9d,0x76]

xvfrintrz.d $xr29, $xr5
# CHECK-INST: xvfrintrz.d $xr29, $xr5
# CHECK-ENCODING: encoding: [0xbd,0x68,0x9d,0x76]

xvfrintrp.s $xr26, $xr16
# CHECK-INST: xvfrintrp.s $xr26, $xr16
# CHECK-ENCODING: encoding: [0x1a,0x56,0x9d,0x76]

xvfrintrp.d $xr1, $xr28
# CHECK-INST: xvfrintrp.d $xr1, $xr28
# CHECK-ENCODING: encoding: [0x81,0x5b,0x9d,0x76]

xvfrintrm.s $xr27, $xr13
# CHECK-INST: xvfrintrm.s $xr27, $xr13
# CHECK-ENCODING: encoding: [0xbb,0x45,0x9d,0x76]

xvfrintrm.d $xr14, $xr27
# CHECK-INST: xvfrintrm.d $xr14, $xr27
# CHECK-ENCODING: encoding: [0x6e,0x4b,0x9d,0x76]

xvfrint.s $xr21, $xr24
# CHECK-INST: xvfrint.s $xr21, $xr24
# CHECK-ENCODING: encoding: [0x15,0x37,0x9d,0x76]

xvfrint.d $xr31, $xr18
# CHECK-INST: xvfrint.d $xr31, $xr18
# CHECK-ENCODING: encoding: [0x5f,0x3a,0x9d,0x76]
