# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvadda.b $xr10, $xr24, $xr27
# CHECK-INST: xvadda.b $xr10, $xr24, $xr27
# CHECK-ENCODING: encoding: [0x0a,0x6f,0x5c,0x74]

xvadda.h $xr0, $xr28, $xr29
# CHECK-INST: xvadda.h $xr0, $xr28, $xr29
# CHECK-ENCODING: encoding: [0x80,0xf7,0x5c,0x74]

xvadda.w $xr31, $xr9, $xr9
# CHECK-INST: xvadda.w $xr31, $xr9, $xr9
# CHECK-ENCODING: encoding: [0x3f,0x25,0x5d,0x74]

xvadda.d $xr10, $xr1, $xr25
# CHECK-INST: xvadda.d $xr10, $xr1, $xr25
# CHECK-ENCODING: encoding: [0x2a,0xe4,0x5d,0x74]
