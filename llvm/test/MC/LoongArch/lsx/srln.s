# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsrln.b.h $vr29, $vr28, $vr27
# CHECK-INST: vsrln.b.h $vr29, $vr28, $vr27
# CHECK-ENCODING: encoding: [0x9d,0xef,0xf4,0x70]

vsrln.h.w $vr18, $vr17, $vr0
# CHECK-INST: vsrln.h.w $vr18, $vr17, $vr0
# CHECK-ENCODING: encoding: [0x32,0x02,0xf5,0x70]

vsrln.w.d $vr16, $vr5, $vr19
# CHECK-INST: vsrln.w.d $vr16, $vr5, $vr19
# CHECK-ENCODING: encoding: [0xb0,0xcc,0xf5,0x70]
