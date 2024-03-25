# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vld $vr0, $r12, -536
# CHECK-INST: vld $vr0, $t0, -536
# CHECK-ENCODING: encoding: [0x80,0xa1,0x37,0x2c]

vldx $vr21, $r14, $r20
# CHECK-INST: vldx $vr21, $t2, $t8
# CHECK-ENCODING: encoding: [0xd5,0x51,0x40,0x38]
