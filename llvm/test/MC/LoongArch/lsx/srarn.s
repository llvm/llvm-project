# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsrarn.b.h $vr19, $vr23, $vr21
# CHECK-INST: vsrarn.b.h $vr19, $vr23, $vr21
# CHECK-ENCODING: encoding: [0xf3,0xd6,0xfa,0x70]

vsrarn.h.w $vr18, $vr6, $vr7
# CHECK-INST: vsrarn.h.w $vr18, $vr6, $vr7
# CHECK-ENCODING: encoding: [0xd2,0x1c,0xfb,0x70]

vsrarn.w.d $vr2, $vr11, $vr5
# CHECK-INST: vsrarn.w.d $vr2, $vr11, $vr5
# CHECK-ENCODING: encoding: [0x62,0x95,0xfb,0x70]
