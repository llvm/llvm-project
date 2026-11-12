# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vclz.b $vr22, $vr14
# CHECK-INST: vclz.b $vr22, $vr14
# CHECK-ENCODING: encoding: [0xd6,0x11,0x9c,0x72]

vclz.h $vr16, $vr0
# CHECK-INST: vclz.h $vr16, $vr0
# CHECK-ENCODING: encoding: [0x10,0x14,0x9c,0x72]

vclz.w $vr19, $vr19
# CHECK-INST: vclz.w $vr19, $vr19
# CHECK-ENCODING: encoding: [0x73,0x1a,0x9c,0x72]

vclz.d $vr27, $vr14
# CHECK-INST: vclz.d $vr27, $vr14
# CHECK-ENCODING: encoding: [0xdb,0x1d,0x9c,0x72]
