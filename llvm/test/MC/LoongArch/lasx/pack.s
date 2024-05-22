# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvpackev.b $xr21, $xr2, $xr8
# CHECK-INST: xvpackev.b $xr21, $xr2, $xr8
# CHECK-ENCODING: encoding: [0x55,0x20,0x16,0x75]

xvpackev.h $xr8, $xr18, $xr6
# CHECK-INST: xvpackev.h $xr8, $xr18, $xr6
# CHECK-ENCODING: encoding: [0x48,0x9a,0x16,0x75]

xvpackev.w $xr0, $xr6, $xr30
# CHECK-INST: xvpackev.w $xr0, $xr6, $xr30
# CHECK-ENCODING: encoding: [0xc0,0x78,0x17,0x75]

xvpackev.d $xr0, $xr9, $xr4
# CHECK-INST: xvpackev.d $xr0, $xr9, $xr4
# CHECK-ENCODING: encoding: [0x20,0x91,0x17,0x75]

xvpackod.b $xr28, $xr29, $xr31
# CHECK-INST: xvpackod.b $xr28, $xr29, $xr31
# CHECK-ENCODING: encoding: [0xbc,0x7f,0x18,0x75]

xvpackod.h $xr14, $xr10, $xr6
# CHECK-INST: xvpackod.h $xr14, $xr10, $xr6
# CHECK-ENCODING: encoding: [0x4e,0x99,0x18,0x75]

xvpackod.w $xr22, $xr21, $xr2
# CHECK-INST: xvpackod.w $xr22, $xr21, $xr2
# CHECK-ENCODING: encoding: [0xb6,0x0a,0x19,0x75]

xvpackod.d $xr18, $xr9, $xr2
# CHECK-INST: xvpackod.d $xr18, $xr9, $xr2
# CHECK-ENCODING: encoding: [0x32,0x89,0x19,0x75]
