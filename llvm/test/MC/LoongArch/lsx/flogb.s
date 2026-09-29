# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vflogb.s $vr12, $vr20
# CHECK-INST: vflogb.s $vr12, $vr20
# CHECK-ENCODING: encoding: [0x8c,0xc6,0x9c,0x72]

vflogb.d $vr3, $vr29
# CHECK-INST: vflogb.d $vr3, $vr29
# CHECK-ENCODING: encoding: [0xa3,0xcb,0x9c,0x72]
