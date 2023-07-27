# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vst $vr9, $r17, 1700
# CHECK-INST: vst $vr9, $t5, 1700
# CHECK-ENCODING: encoding: [0x29,0x92,0x5a,0x2c]

vstx $vr23, $r17, $r31
# CHECK-INST: vstx $vr23, $t5, $s8
# CHECK-ENCODING: encoding: [0x37,0x7e,0x44,0x38]
