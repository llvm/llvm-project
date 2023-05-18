# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfcvt.h.s $vr30, $vr1, $vr30
# CHECK-INST: vfcvt.h.s $vr30, $vr1, $vr30
# CHECK-ENCODING: encoding: [0x3e,0x78,0x46,0x71]

vfcvt.s.d $vr27, $vr11, $vr4
# CHECK-INST: vfcvt.s.d $vr27, $vr11, $vr4
# CHECK-ENCODING: encoding: [0x7b,0x91,0x46,0x71]
