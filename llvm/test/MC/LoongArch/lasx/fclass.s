# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfclass.s $xr3, $xr7
# CHECK-INST: xvfclass.s $xr3, $xr7
# CHECK-ENCODING: encoding: [0xe3,0xd4,0x9c,0x76]

xvfclass.d $xr22, $xr10
# CHECK-INST: xvfclass.d $xr22, $xr10
# CHECK-ENCODING: encoding: [0x56,0xd9,0x9c,0x76]
