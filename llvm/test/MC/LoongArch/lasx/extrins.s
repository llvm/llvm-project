# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvextrins.b $xr30, $xr23, 252
# CHECK-INST: xvextrins.b $xr30, $xr23, 252
# CHECK-ENCODING: encoding: [0xfe,0xf2,0x8f,0x77]

xvextrins.h $xr0, $xr13, 200
# CHECK-INST: xvextrins.h $xr0, $xr13, 200
# CHECK-ENCODING: encoding: [0xa0,0x21,0x8b,0x77]

xvextrins.w $xr14, $xr21, 152
# CHECK-INST: xvextrins.w $xr14, $xr21, 152
# CHECK-ENCODING: encoding: [0xae,0x62,0x86,0x77]

xvextrins.d $xr31, $xr30, 135
# CHECK-INST: xvextrins.d $xr31, $xr30, 135
# CHECK-ENCODING: encoding: [0xdf,0x1f,0x82,0x77]
