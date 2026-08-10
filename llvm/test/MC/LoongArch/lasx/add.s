# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvadd.b $xr20, $xr19, $xr5
# CHECK-INST: xvadd.b $xr20, $xr19, $xr5
# CHECK-ENCODING: encoding: [0x74,0x16,0x0a,0x74]

xvadd.h $xr24, $xr7, $xr14
# CHECK-INST: xvadd.h $xr24, $xr7, $xr14
# CHECK-ENCODING: encoding: [0xf8,0xb8,0x0a,0x74]

xvadd.w $xr19, $xr1, $xr21
# CHECK-INST: xvadd.w $xr19, $xr1, $xr21
# CHECK-ENCODING: encoding: [0x33,0x54,0x0b,0x74]

xvadd.d $xr19, $xr6, $xr13
# CHECK-INST: xvadd.d $xr19, $xr6, $xr13
# CHECK-ENCODING: encoding: [0xd3,0xb4,0x0b,0x74]

xvadd.q $xr4, $xr28, $xr6
# CHECK-INST: xvadd.q $xr4, $xr28, $xr6
# CHECK-ENCODING: encoding: [0x84,0x1b,0x2d,0x75]
