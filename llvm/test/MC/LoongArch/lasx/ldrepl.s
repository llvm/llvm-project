# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvldrepl.b $xr19, $r21, 1892
# CHECK-INST: xvldrepl.b $xr19, $r21, 1892
# CHECK-ENCODING: encoding: [0xb3,0x92,0x9d,0x32]

xvldrepl.h $xr0, $r17, 1762
# CHECK-INST: xvldrepl.h $xr0, $t5, 1762
# CHECK-ENCODING: encoding: [0x20,0xc6,0x4d,0x32]

xvldrepl.w $xr11, $r26, -1524
# CHECK-INST: xvldrepl.w $xr11, $s3, -1524
# CHECK-ENCODING: encoding: [0x4b,0x0f,0x2a,0x32]

xvldrepl.d $xr28, $r12, 1976
# CHECK-INST: xvldrepl.d $xr28, $t0, 1976
# CHECK-ENCODING: encoding: [0x9c,0xdd,0x13,0x32]
