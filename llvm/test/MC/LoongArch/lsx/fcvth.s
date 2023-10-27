# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfcvth.s.h $vr7, $vr30
# CHECK-INST: vfcvth.s.h $vr7, $vr30
# CHECK-ENCODING: encoding: [0xc7,0xef,0x9d,0x72]

vfcvth.d.s $vr15, $vr14
# CHECK-INST: vfcvth.d.s $vr15, $vr14
# CHECK-ENCODING: encoding: [0xcf,0xf5,0x9d,0x72]
