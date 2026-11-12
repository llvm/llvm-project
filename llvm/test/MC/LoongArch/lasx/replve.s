# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvreplve.b $xr20, $xr16, $r11
# CHECK-INST: xvreplve.b $xr20, $xr16, $a7
# CHECK-ENCODING: encoding: [0x14,0x2e,0x22,0x75]

xvreplve.h $xr0, $xr21, $r24
# CHECK-INST: xvreplve.h $xr0, $xr21, $s1
# CHECK-ENCODING: encoding: [0xa0,0xe2,0x22,0x75]

xvreplve.w $xr20, $xr18, $r18
# CHECK-INST: xvreplve.w $xr20, $xr18, $t6
# CHECK-ENCODING: encoding: [0x54,0x4a,0x23,0x75]

xvreplve.d $xr4, $xr3, $r23
# CHECK-INST: xvreplve.d $xr4, $xr3, $s0
# CHECK-ENCODING: encoding: [0x64,0xdc,0x23,0x75]
