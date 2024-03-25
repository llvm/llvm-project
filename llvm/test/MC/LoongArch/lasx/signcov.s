# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsigncov.b $xr1, $xr24, $xr13
# CHECK-INST: xvsigncov.b $xr1, $xr24, $xr13
# CHECK-ENCODING: encoding: [0x01,0x37,0x2e,0x75]

xvsigncov.h $xr8, $xr23, $xr14
# CHECK-INST: xvsigncov.h $xr8, $xr23, $xr14
# CHECK-ENCODING: encoding: [0xe8,0xba,0x2e,0x75]

xvsigncov.w $xr3, $xr25, $xr10
# CHECK-INST: xvsigncov.w $xr3, $xr25, $xr10
# CHECK-ENCODING: encoding: [0x23,0x2b,0x2f,0x75]

xvsigncov.d $xr26, $xr17, $xr31
# CHECK-INST: xvsigncov.d $xr26, $xr17, $xr31
# CHECK-ENCODING: encoding: [0x3a,0xfe,0x2f,0x75]
