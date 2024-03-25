# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vinsgr2vr.b $vr23, $r20, 2
# CHECK-INST: vinsgr2vr.b $vr23, $t8, 2
# CHECK-ENCODING: encoding: [0x97,0x8a,0xeb,0x72]

vinsgr2vr.h $vr7, $r5, 7
# CHECK-INST: vinsgr2vr.h $vr7, $a1, 7
# CHECK-ENCODING: encoding: [0xa7,0xdc,0xeb,0x72]

vinsgr2vr.w $vr8, $r6, 2
# CHECK-INST: vinsgr2vr.w $vr8, $a2, 2
# CHECK-ENCODING: encoding: [0xc8,0xe8,0xeb,0x72]

vinsgr2vr.d $vr17, $r24, 1
# CHECK-INST: vinsgr2vr.d $vr17, $s1, 1
# CHECK-ENCODING: encoding: [0x11,0xf7,0xeb,0x72]
