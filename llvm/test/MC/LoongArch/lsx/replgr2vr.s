# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vreplgr2vr.b $vr30, $r27
# CHECK-INST: vreplgr2vr.b $vr30, $s4
# CHECK-ENCODING: encoding: [0x7e,0x03,0x9f,0x72]

vreplgr2vr.h $vr6, $r1
# CHECK-INST: vreplgr2vr.h $vr6, $ra
# CHECK-ENCODING: encoding: [0x26,0x04,0x9f,0x72]

vreplgr2vr.w $vr23, $r9
# CHECK-INST: vreplgr2vr.w $vr23, $a5
# CHECK-ENCODING: encoding: [0x37,0x09,0x9f,0x72]

vreplgr2vr.d $vr17, $r14
# CHECK-INST: vreplgr2vr.d $vr17, $t2
# CHECK-ENCODING: encoding: [0xd1,0x0d,0x9f,0x72]
