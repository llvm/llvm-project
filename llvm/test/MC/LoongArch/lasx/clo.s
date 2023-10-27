# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvclo.b $xr9, $xr12
# CHECK-INST: xvclo.b $xr9, $xr12
# CHECK-ENCODING: encoding: [0x89,0x01,0x9c,0x76]

xvclo.h $xr16, $xr14
# CHECK-INST: xvclo.h $xr16, $xr14
# CHECK-ENCODING: encoding: [0xd0,0x05,0x9c,0x76]

xvclo.w $xr30, $xr18
# CHECK-INST: xvclo.w $xr30, $xr18
# CHECK-ENCODING: encoding: [0x5e,0x0a,0x9c,0x76]

xvclo.d $xr31, $xr5
# CHECK-INST: xvclo.d $xr31, $xr5
# CHECK-ENCODING: encoding: [0xbf,0x0c,0x9c,0x76]
