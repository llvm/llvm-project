# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmskltz.b $vr17, $vr20
# CHECK-INST: vmskltz.b $vr17, $vr20
# CHECK-ENCODING: encoding: [0x91,0x42,0x9c,0x72]

vmskltz.h $vr23, $vr1
# CHECK-INST: vmskltz.h $vr23, $vr1
# CHECK-ENCODING: encoding: [0x37,0x44,0x9c,0x72]

vmskltz.w $vr3, $vr16
# CHECK-INST: vmskltz.w $vr3, $vr16
# CHECK-ENCODING: encoding: [0x03,0x4a,0x9c,0x72]

vmskltz.d $vr1, $vr26
# CHECK-INST: vmskltz.d $vr1, $vr26
# CHECK-ENCODING: encoding: [0x41,0x4f,0x9c,0x72]
