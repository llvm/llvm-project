# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvfmul.s $xr9, $xr14, $xr30
# CHECK-INST: xvfmul.s $xr9, $xr14, $xr30
# CHECK-ENCODING: encoding: [0xc9,0xf9,0x38,0x75]

xvfmul.d $xr28, $xr26, $xr19
# CHECK-INST: xvfmul.d $xr28, $xr26, $xr19
# CHECK-ENCODING: encoding: [0x5c,0x4f,0x39,0x75]
