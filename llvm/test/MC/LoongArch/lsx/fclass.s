# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfclass.s $vr24, $vr26
# CHECK-INST: vfclass.s $vr24, $vr26
# CHECK-ENCODING: encoding: [0x58,0xd7,0x9c,0x72]

vfclass.d $vr8, $vr17
# CHECK-INST: vfclass.d $vr8, $vr17
# CHECK-ENCODING: encoding: [0x28,0xda,0x9c,0x72]
