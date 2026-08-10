# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsrani.b.h $vr3, $vr0, 9
# CHECK-INST: vsrani.b.h $vr3, $vr0, 9
# CHECK-ENCODING: encoding: [0x03,0x64,0x58,0x73]

vsrani.h.w $vr4, $vr3, 26
# CHECK-INST: vsrani.h.w $vr4, $vr3, 26
# CHECK-ENCODING: encoding: [0x64,0xe8,0x58,0x73]

vsrani.w.d $vr8, $vr27, 52
# CHECK-INST: vsrani.w.d $vr8, $vr27, 52
# CHECK-ENCODING: encoding: [0x68,0xd3,0x59,0x73]

vsrani.d.q $vr21, $vr24, 28
# CHECK-INST: vsrani.d.q $vr21, $vr24, 28
# CHECK-ENCODING: encoding: [0x15,0x73,0x5a,0x73]
