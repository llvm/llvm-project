# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vclo.b $vr2, $vr0
# CHECK-INST: vclo.b $vr2, $vr0
# CHECK-ENCODING: encoding: [0x02,0x00,0x9c,0x72]

vclo.h $vr23, $vr31
# CHECK-INST: vclo.h $vr23, $vr31
# CHECK-ENCODING: encoding: [0xf7,0x07,0x9c,0x72]

vclo.w $vr7, $vr28
# CHECK-INST: vclo.w $vr7, $vr28
# CHECK-ENCODING: encoding: [0x87,0x0b,0x9c,0x72]

vclo.d $vr5, $vr11
# CHECK-INST: vclo.d $vr5, $vr11
# CHECK-ENCODING: encoding: [0x65,0x0d,0x9c,0x72]
