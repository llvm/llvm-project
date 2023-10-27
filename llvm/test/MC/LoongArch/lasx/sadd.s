# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsadd.b $xr27, $xr30, $xr22
# CHECK-INST: xvsadd.b $xr27, $xr30, $xr22
# CHECK-ENCODING: encoding: [0xdb,0x5b,0x46,0x74]

xvsadd.h $xr29, $xr0, $xr1
# CHECK-INST: xvsadd.h $xr29, $xr0, $xr1
# CHECK-ENCODING: encoding: [0x1d,0x84,0x46,0x74]

xvsadd.w $xr22, $xr28, $xr31
# CHECK-INST: xvsadd.w $xr22, $xr28, $xr31
# CHECK-ENCODING: encoding: [0x96,0x7f,0x47,0x74]

xvsadd.d $xr5, $xr18, $xr26
# CHECK-INST: xvsadd.d $xr5, $xr18, $xr26
# CHECK-ENCODING: encoding: [0x45,0xea,0x47,0x74]

xvsadd.bu $xr29, $xr20, $xr28
# CHECK-INST: xvsadd.bu $xr29, $xr20, $xr28
# CHECK-ENCODING: encoding: [0x9d,0x72,0x4a,0x74]

xvsadd.hu $xr7, $xr16, $xr6
# CHECK-INST: xvsadd.hu $xr7, $xr16, $xr6
# CHECK-ENCODING: encoding: [0x07,0x9a,0x4a,0x74]

xvsadd.wu $xr2, $xr10, $xr15
# CHECK-INST: xvsadd.wu $xr2, $xr10, $xr15
# CHECK-ENCODING: encoding: [0x42,0x3d,0x4b,0x74]

xvsadd.du $xr18, $xr24, $xr14
# CHECK-INST: xvsadd.du $xr18, $xr24, $xr14
# CHECK-ENCODING: encoding: [0x12,0xbb,0x4b,0x74]
