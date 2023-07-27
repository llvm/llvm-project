# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfcvtl.s.h $vr26, $vr23
# CHECK-INST: vfcvtl.s.h $vr26, $vr23
# CHECK-ENCODING: encoding: [0xfa,0xea,0x9d,0x72]

vfcvtl.d.s $vr3, $vr7
# CHECK-INST: vfcvtl.d.s $vr3, $vr7
# CHECK-ENCODING: encoding: [0xe3,0xf0,0x9d,0x72]
