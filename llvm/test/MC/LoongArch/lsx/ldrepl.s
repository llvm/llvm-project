# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vldrepl.b $vr3, $r3, -1553
# CHECK-INST: vldrepl.b $vr3, $sp, -1553
# CHECK-ENCODING: encoding: [0x63,0xbc,0xa7,0x30]

vldrepl.h $vr23, $r22, 172
# CHECK-INST: vldrepl.h $vr23, $fp, 172
# CHECK-ENCODING: encoding: [0xd7,0x5a,0x41,0x30]

vldrepl.w $vr12, $r27, -1304
# CHECK-INST: vldrepl.w $vr12, $s4, -1304
# CHECK-ENCODING: encoding: [0x6c,0xeb,0x2a,0x30]

vldrepl.d $vr7, $r31, -1376
# CHECK-INST: vldrepl.d $vr7, $s8, -1376
# CHECK-ENCODING: encoding: [0xe7,0x53,0x15,0x30]
