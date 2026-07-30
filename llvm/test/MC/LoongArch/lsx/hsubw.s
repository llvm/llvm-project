# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vhsubw.h.b $vr24, $vr26, $vr16
# CHECK-INST: vhsubw.h.b $vr24, $vr26, $vr16
# CHECK-ENCODING: encoding: [0x58,0x43,0x56,0x70]

vhsubw.w.h $vr5, $vr28, $vr12
# CHECK-INST: vhsubw.w.h $vr5, $vr28, $vr12
# CHECK-ENCODING: encoding: [0x85,0xb3,0x56,0x70]

vhsubw.d.w $vr8, $vr5, $vr22
# CHECK-INST: vhsubw.d.w $vr8, $vr5, $vr22
# CHECK-ENCODING: encoding: [0xa8,0x58,0x57,0x70]

vhsubw.q.d $vr21, $vr16, $vr14
# CHECK-INST: vhsubw.q.d $vr21, $vr16, $vr14
# CHECK-ENCODING: encoding: [0x15,0xba,0x57,0x70]

vhsubw.hu.bu $vr12, $vr31, $vr30
# CHECK-INST: vhsubw.hu.bu $vr12, $vr31, $vr30
# CHECK-ENCODING: encoding: [0xec,0x7b,0x5a,0x70]

vhsubw.wu.hu $vr18, $vr13, $vr31
# CHECK-INST: vhsubw.wu.hu $vr18, $vr13, $vr31
# CHECK-ENCODING: encoding: [0xb2,0xfd,0x5a,0x70]

vhsubw.du.wu $vr0, $vr1, $vr2
# CHECK-INST: vhsubw.du.wu $vr0, $vr1, $vr2
# CHECK-ENCODING: encoding: [0x20,0x08,0x5b,0x70]

vhsubw.qu.du $vr30, $vr31, $vr5
# CHECK-INST: vhsubw.qu.du $vr30, $vr31, $vr5
# CHECK-ENCODING: encoding: [0xfe,0x97,0x5b,0x70]
