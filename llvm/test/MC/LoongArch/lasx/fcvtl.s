# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfcvtl.s.h $xr16, $xr14
# CHECK-INST: xvfcvtl.s.h $xr16, $xr14
# CHECK-ENCODING: encoding: [0xd0,0xe9,0x9d,0x76]

xvfcvtl.d.s $xr24, $xr5
# CHECK-INST: xvfcvtl.d.s $xr24, $xr5
# CHECK-ENCODING: encoding: [0xb8,0xf0,0x9d,0x76]
