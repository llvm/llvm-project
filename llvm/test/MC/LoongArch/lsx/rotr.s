# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vrotr.b $vr15, $vr25, $vr30
# CHECK-INST: vrotr.b $vr15, $vr25, $vr30
# CHECK-ENCODING: encoding: [0x2f,0x7b,0xee,0x70]

vrotr.h $vr5, $vr23, $vr14
# CHECK-INST: vrotr.h $vr5, $vr23, $vr14
# CHECK-ENCODING: encoding: [0xe5,0xba,0xee,0x70]

vrotr.w $vr27, $vr0, $vr7
# CHECK-INST: vrotr.w $vr27, $vr0, $vr7
# CHECK-ENCODING: encoding: [0x1b,0x1c,0xef,0x70]

vrotr.d $vr2, $vr3, $vr21
# CHECK-INST: vrotr.d $vr2, $vr3, $vr21
# CHECK-ENCODING: encoding: [0x62,0xd4,0xef,0x70]

vrotri.b $vr17, $vr22, 5
# CHECK-INST: vrotri.b $vr17, $vr22, 5
# CHECK-ENCODING: encoding: [0xd1,0x36,0xa0,0x72]

vrotri.h $vr27, $vr20, 10
# CHECK-INST: vrotri.h $vr27, $vr20, 10
# CHECK-ENCODING: encoding: [0x9b,0x6a,0xa0,0x72]

vrotri.w $vr21, $vr24, 14
# CHECK-INST: vrotri.w $vr21, $vr24, 14
# CHECK-ENCODING: encoding: [0x15,0xbb,0xa0,0x72]

vrotri.d $vr25, $vr23, 14
# CHECK-INST: vrotri.d $vr25, $vr23, 14
# CHECK-ENCODING: encoding: [0xf9,0x3a,0xa1,0x72]
