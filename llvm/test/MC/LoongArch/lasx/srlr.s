# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsrlr.b $xr18, $xr11, $xr5
# CHECK-INST: xvsrlr.b $xr18, $xr11, $xr5
# CHECK-ENCODING: encoding: [0x72,0x15,0xf0,0x74]

xvsrlr.h $xr31, $xr5, $xr21
# CHECK-INST: xvsrlr.h $xr31, $xr5, $xr21
# CHECK-ENCODING: encoding: [0xbf,0xd4,0xf0,0x74]

xvsrlr.w $xr7, $xr5, $xr1
# CHECK-INST: xvsrlr.w $xr7, $xr5, $xr1
# CHECK-ENCODING: encoding: [0xa7,0x04,0xf1,0x74]

xvsrlr.d $xr4, $xr27, $xr7
# CHECK-INST: xvsrlr.d $xr4, $xr27, $xr7
# CHECK-ENCODING: encoding: [0x64,0x9f,0xf1,0x74]

xvsrlri.b $xr29, $xr30, 4
# CHECK-INST: xvsrlri.b $xr29, $xr30, 4
# CHECK-ENCODING: encoding: [0xdd,0x33,0xa4,0x76]

xvsrlri.h $xr16, $xr6, 14
# CHECK-INST: xvsrlri.h $xr16, $xr6, 14
# CHECK-ENCODING: encoding: [0xd0,0x78,0xa4,0x76]

xvsrlri.w $xr24, $xr10, 28
# CHECK-INST: xvsrlri.w $xr24, $xr10, 28
# CHECK-ENCODING: encoding: [0x58,0xf1,0xa4,0x76]

xvsrlri.d $xr20, $xr20, 52
# CHECK-INST: xvsrlri.d $xr20, $xr20, 52
# CHECK-ENCODING: encoding: [0x94,0xd2,0xa5,0x76]
