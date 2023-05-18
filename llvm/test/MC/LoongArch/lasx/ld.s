# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

xvld $xr3, $r3, -658
# CHECK-INST: xvld $xr3, $sp, -658
# CHECK-ENCODING: encoding: [0x63,0xb8,0xb5,0x2c]

xvldx $xr23, $r9, $r14
# CHECK-INST: xvldx $xr23, $a5, $t2
# CHECK-ENCODING: encoding: [0x37,0x39,0x48,0x38]
