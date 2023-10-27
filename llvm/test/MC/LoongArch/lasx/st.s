# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvst $xr14, $r12, 943
# CHECK-INST: xvst $xr14, $t0, 943
# CHECK-ENCODING: encoding: [0x8e,0xbd,0xce,0x2c]

xvstx $xr7, $r9, $r21
# CHECK-INST: xvstx $xr7, $a5, $r21
# CHECK-ENCODING: encoding: [0x27,0x55,0x4c,0x38]
