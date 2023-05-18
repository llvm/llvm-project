# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfrstp.b $vr30, $vr25, $vr1
# CHECK-INST: vfrstp.b $vr30, $vr25, $vr1
# CHECK-ENCODING: encoding: [0x3e,0x07,0x2b,0x71]

vfrstp.h $vr22, $vr26, $vr21
# CHECK-INST: vfrstp.h $vr22, $vr26, $vr21
# CHECK-ENCODING: encoding: [0x56,0xd7,0x2b,0x71]

vfrstpi.b $vr12, $vr8, 28
# CHECK-INST: vfrstpi.b $vr12, $vr8, 28
# CHECK-ENCODING: encoding: [0x0c,0x71,0x9a,0x72]

vfrstpi.h $vr5, $vr28, 29
# CHECK-INST: vfrstpi.h $vr5, $vr28, 29
# CHECK-ENCODING: encoding: [0x85,0xf7,0x9a,0x72]
