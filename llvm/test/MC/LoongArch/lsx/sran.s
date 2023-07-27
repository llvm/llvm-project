# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsran.b.h $vr25, $vr2, $vr31
# CHECK-INST: vsran.b.h $vr25, $vr2, $vr31
# CHECK-ENCODING: encoding: [0x59,0xfc,0xf6,0x70]

vsran.h.w $vr31, $vr10, $vr3
# CHECK-INST: vsran.h.w $vr31, $vr10, $vr3
# CHECK-ENCODING: encoding: [0x5f,0x0d,0xf7,0x70]

vsran.w.d $vr8, $vr3, $vr12
# CHECK-INST: vsran.w.d $vr8, $vr3, $vr12
# CHECK-ENCODING: encoding: [0x68,0xb0,0xf7,0x70]
