# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvsrlrn.b.h $xr4, $xr25, $xr26
# CHECK-INST: xvsrlrn.b.h $xr4, $xr25, $xr26
# CHECK-ENCODING: encoding: [0x24,0xeb,0xf8,0x74]

xvsrlrn.h.w $xr17, $xr5, $xr1
# CHECK-INST: xvsrlrn.h.w $xr17, $xr5, $xr1
# CHECK-ENCODING: encoding: [0xb1,0x04,0xf9,0x74]

xvsrlrn.w.d $xr29, $xr1, $xr17
# CHECK-INST: xvsrlrn.w.d $xr29, $xr1, $xr17
# CHECK-ENCODING: encoding: [0x3d,0xc4,0xf9,0x74]
