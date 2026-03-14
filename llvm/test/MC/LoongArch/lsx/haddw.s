# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vhaddw.h.b $vr3, $vr14, $vr11
# CHECK-INST: vhaddw.h.b $vr3, $vr14, $vr11
# CHECK-ENCODING: encoding: [0xc3,0x2d,0x54,0x70]

vhaddw.w.h $vr3, $vr9, $vr9
# CHECK-INST: vhaddw.w.h $vr3, $vr9, $vr9
# CHECK-ENCODING: encoding: [0x23,0xa5,0x54,0x70]

vhaddw.d.w $vr7, $vr26, $vr6
# CHECK-INST: vhaddw.d.w $vr7, $vr26, $vr6
# CHECK-ENCODING: encoding: [0x47,0x1b,0x55,0x70]

vhaddw.q.d $vr22, $vr25, $vr19
# CHECK-INST: vhaddw.q.d $vr22, $vr25, $vr19
# CHECK-ENCODING: encoding: [0x36,0xcf,0x55,0x70]

vhaddw.hu.bu $vr8, $vr21, $vr21
# CHECK-INST: vhaddw.hu.bu $vr8, $vr21, $vr21
# CHECK-ENCODING: encoding: [0xa8,0x56,0x58,0x70]

vhaddw.wu.hu $vr23, $vr23, $vr20
# CHECK-INST: vhaddw.wu.hu $vr23, $vr23, $vr20
# CHECK-ENCODING: encoding: [0xf7,0xd2,0x58,0x70]

vhaddw.du.wu $vr13, $vr7, $vr6
# CHECK-INST: vhaddw.du.wu $vr13, $vr7, $vr6
# CHECK-ENCODING: encoding: [0xed,0x18,0x59,0x70]

vhaddw.qu.du $vr19, $vr12, $vr6
# CHECK-INST: vhaddw.qu.du $vr19, $vr12, $vr6
# CHECK-ENCODING: encoding: [0x93,0x99,0x59,0x70]
