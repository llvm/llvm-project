# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vssrarn.b.h $vr27, $vr29, $vr23
# CHECK-INST: vssrarn.b.h $vr27, $vr29, $vr23
# CHECK-ENCODING: encoding: [0xbb,0xdf,0x02,0x71]

vssrarn.h.w $vr13, $vr17, $vr0
# CHECK-INST: vssrarn.h.w $vr13, $vr17, $vr0
# CHECK-ENCODING: encoding: [0x2d,0x02,0x03,0x71]

vssrarn.w.d $vr5, $vr11, $vr16
# CHECK-INST: vssrarn.w.d $vr5, $vr11, $vr16
# CHECK-ENCODING: encoding: [0x65,0xc1,0x03,0x71]

vssrarn.bu.h $vr18, $vr10, $vr13
# CHECK-INST: vssrarn.bu.h $vr18, $vr10, $vr13
# CHECK-ENCODING: encoding: [0x52,0xb5,0x0a,0x71]

vssrarn.hu.w $vr5, $vr25, $vr16
# CHECK-INST: vssrarn.hu.w $vr5, $vr25, $vr16
# CHECK-ENCODING: encoding: [0x25,0x43,0x0b,0x71]

vssrarn.wu.d $vr6, $vr23, $vr30
# CHECK-INST: vssrarn.wu.d $vr6, $vr23, $vr30
# CHECK-ENCODING: encoding: [0xe6,0xfa,0x0b,0x71]
