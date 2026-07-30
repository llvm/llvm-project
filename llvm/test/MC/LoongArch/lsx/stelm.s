# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vstelm.b $vr22, $r31, -90, 12
# CHECK-INST: vstelm.b $vr22, $s8, -90, 12
# CHECK-ENCODING: encoding: [0xf6,0x9b,0xb2,0x31]

vstelm.h $vr28, $r2, 48, 7
# CHECK-INST: vstelm.h $vr28, $tp, 48, 7
# CHECK-ENCODING: encoding: [0x5c,0x60,0x5c,0x31]

vstelm.w $vr18, $r12, -40, 2
# CHECK-INST: vstelm.w $vr18, $t0, -40, 2
# CHECK-ENCODING: encoding: [0x92,0xd9,0x2b,0x31]

vstelm.d $vr4, $r23, -248, 1
# CHECK-INST: vstelm.d $vr4, $s0, -248, 1
# CHECK-ENCODING: encoding: [0xe4,0x86,0x17,0x31]
