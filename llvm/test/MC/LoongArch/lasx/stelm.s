# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvstelm.b $xr20, $r2, -105, 10
# CHECK-INST: xvstelm.b $xr20, $tp, -105, 10
# CHECK-ENCODING: encoding: [0x54,0x5c,0xaa,0x33]

xvstelm.h $xr8, $r1, 160, 4
# CHECK-INST: xvstelm.h $xr8, $ra, 160, 4
# CHECK-ENCODING: encoding: [0x28,0x40,0x51,0x33]

xvstelm.w $xr19, $r18, 412, 0
# CHECK-INST: xvstelm.w $xr19, $t6, 412, 0
# CHECK-ENCODING: encoding: [0x53,0x9e,0x21,0x33]

xvstelm.d $xr22, $r30, 960, 3
# CHECK-INST: xvstelm.d $xr22, $s7, 960, 3
# CHECK-ENCODING: encoding: [0xd6,0xe3,0x1d,0x33]
