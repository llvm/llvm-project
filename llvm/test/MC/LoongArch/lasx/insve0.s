# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvinsve0.w $xr6, $xr1, 7
# CHECK-INST: xvinsve0.w $xr6, $xr1, 7
# CHECK-ENCODING: encoding: [0x26,0xdc,0xff,0x76]

xvinsve0.d $xr28, $xr1, 0
# CHECK-INST: xvinsve0.d $xr28, $xr1, 0
# CHECK-ENCODING: encoding: [0x3c,0xe0,0xff,0x76]
