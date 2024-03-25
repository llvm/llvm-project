# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfadd.s $xr6, $xr21, $xr15
# CHECK-INST: xvfadd.s $xr6, $xr21, $xr15
# CHECK-ENCODING: encoding: [0xa6,0xbe,0x30,0x75]

xvfadd.d $xr27, $xr8, $xr1
# CHECK-INST: xvfadd.d $xr27, $xr8, $xr1
# CHECK-ENCODING: encoding: [0x1b,0x05,0x31,0x75]
