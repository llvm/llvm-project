# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvssrlrn.b.h $xr8, $xr20, $xr18
# CHECK-INST: xvssrlrn.b.h $xr8, $xr20, $xr18
# CHECK-ENCODING: encoding: [0x88,0xca,0x00,0x75]

xvssrlrn.h.w $xr2, $xr13, $xr19
# CHECK-INST: xvssrlrn.h.w $xr2, $xr13, $xr19
# CHECK-ENCODING: encoding: [0xa2,0x4d,0x01,0x75]

xvssrlrn.w.d $xr24, $xr7, $xr5
# CHECK-INST: xvssrlrn.w.d $xr24, $xr7, $xr5
# CHECK-ENCODING: encoding: [0xf8,0x94,0x01,0x75]

xvssrlrn.bu.h $xr15, $xr23, $xr18
# CHECK-INST: xvssrlrn.bu.h $xr15, $xr23, $xr18
# CHECK-ENCODING: encoding: [0xef,0xca,0x08,0x75]

xvssrlrn.hu.w $xr22, $xr14, $xr16
# CHECK-INST: xvssrlrn.hu.w $xr22, $xr14, $xr16
# CHECK-ENCODING: encoding: [0xd6,0x41,0x09,0x75]

xvssrlrn.wu.d $xr20, $xr28, $xr5
# CHECK-INST: xvssrlrn.wu.d $xr20, $xr28, $xr5
# CHECK-ENCODING: encoding: [0x94,0x97,0x09,0x75]
