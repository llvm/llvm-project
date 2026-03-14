# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vssrln.b.h $vr20, $vr5, $vr20
# CHECK-INST: vssrln.b.h $vr20, $vr5, $vr20
# CHECK-ENCODING: encoding: [0xb4,0xd0,0xfc,0x70]

vssrln.h.w $vr0, $vr21, $vr2
# CHECK-INST: vssrln.h.w $vr0, $vr21, $vr2
# CHECK-ENCODING: encoding: [0xa0,0x0a,0xfd,0x70]

vssrln.w.d $vr16, $vr6, $vr3
# CHECK-INST: vssrln.w.d $vr16, $vr6, $vr3
# CHECK-ENCODING: encoding: [0xd0,0x8c,0xfd,0x70]

vssrln.bu.h $vr6, $vr30, $vr9
# CHECK-INST: vssrln.bu.h $vr6, $vr30, $vr9
# CHECK-ENCODING: encoding: [0xc6,0xa7,0x04,0x71]

vssrln.hu.w $vr2, $vr8, $vr3
# CHECK-INST: vssrln.hu.w $vr2, $vr8, $vr3
# CHECK-ENCODING: encoding: [0x02,0x0d,0x05,0x71]

vssrln.wu.d $vr28, $vr28, $vr5
# CHECK-INST: vssrln.wu.d $vr28, $vr28, $vr5
# CHECK-ENCODING: encoding: [0x9c,0x97,0x05,0x71]
