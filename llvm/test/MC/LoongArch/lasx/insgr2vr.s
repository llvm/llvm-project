# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvinsgr2vr.w $xr25, $r30, 7
# CHECK-INST: xvinsgr2vr.w $xr25, $s7, 7
# CHECK-ENCODING: encoding: [0xd9,0xdf,0xeb,0x76]

xvinsgr2vr.d $xr27, $r21, 1
# CHECK-INST: xvinsgr2vr.d $xr27, $r21, 1
# CHECK-ENCODING: encoding: [0xbb,0xe6,0xeb,0x76]
