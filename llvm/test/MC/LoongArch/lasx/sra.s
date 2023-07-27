# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsra.b $xr11, $xr2, $xr0
# CHECK-INST: xvsra.b $xr11, $xr2, $xr0
# CHECK-ENCODING: encoding: [0x4b,0x00,0xec,0x74]

xvsra.h $xr17, $xr27, $xr6
# CHECK-INST: xvsra.h $xr17, $xr27, $xr6
# CHECK-ENCODING: encoding: [0x71,0x9b,0xec,0x74]

xvsra.w $xr13, $xr12, $xr12
# CHECK-INST: xvsra.w $xr13, $xr12, $xr12
# CHECK-ENCODING: encoding: [0x8d,0x31,0xed,0x74]

xvsra.d $xr6, $xr15, $xr1
# CHECK-INST: xvsra.d $xr6, $xr15, $xr1
# CHECK-ENCODING: encoding: [0xe6,0x85,0xed,0x74]

xvsrai.b $xr16, $xr2, 3
# CHECK-INST: xvsrai.b $xr16, $xr2, 3
# CHECK-ENCODING: encoding: [0x50,0x2c,0x34,0x77]

xvsrai.h $xr14, $xr3, 12
# CHECK-INST: xvsrai.h $xr14, $xr3, 12
# CHECK-ENCODING: encoding: [0x6e,0x70,0x34,0x77]

xvsrai.w $xr17, $xr18, 21
# CHECK-INST: xvsrai.w $xr17, $xr18, 21
# CHECK-ENCODING: encoding: [0x51,0xd6,0x34,0x77]

xvsrai.d $xr10, $xr20, 4
# CHECK-INST: xvsrai.d $xr10, $xr20, 4
# CHECK-ENCODING: encoding: [0x8a,0x12,0x35,0x77]
