# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvshuf.b $xr20, $xr6, $xr11, $xr15
# CHECK-INST: xvshuf.b $xr20, $xr6, $xr11, $xr15
# CHECK-ENCODING: encoding: [0xd4,0xac,0x67,0x0d]

xvshuf.h $xr29, $xr24, $xr1
# CHECK-INST: xvshuf.h $xr29, $xr24, $xr1
# CHECK-ENCODING: encoding: [0x1d,0x87,0x7a,0x75]

xvshuf.w $xr15, $xr24, $xr29
# CHECK-INST: xvshuf.w $xr15, $xr24, $xr29
# CHECK-ENCODING: encoding: [0x0f,0x77,0x7b,0x75]

xvshuf.d $xr27, $xr18, $xr15
# CHECK-INST: xvshuf.d $xr27, $xr18, $xr15
# CHECK-ENCODING: encoding: [0x5b,0xbe,0x7b,0x75]
