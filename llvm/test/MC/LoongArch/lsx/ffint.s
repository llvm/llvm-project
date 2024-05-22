# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vffint.s.w $vr3, $vr0
# CHECK-INST: vffint.s.w $vr3, $vr0
# CHECK-ENCODING: encoding: [0x03,0x00,0x9e,0x72]

vffint.d.l $vr2, $vr15
# CHECK-INST: vffint.d.l $vr2, $vr15
# CHECK-ENCODING: encoding: [0xe2,0x09,0x9e,0x72]

vffint.s.wu $vr5, $vr9
# CHECK-INST: vffint.s.wu $vr5, $vr9
# CHECK-ENCODING: encoding: [0x25,0x05,0x9e,0x72]

vffint.d.lu $vr6, $vr13
# CHECK-INST: vffint.d.lu $vr6, $vr13
# CHECK-ENCODING: encoding: [0xa6,0x0d,0x9e,0x72]

vffintl.d.w $vr26, $vr1
# CHECK-INST: vffintl.d.w $vr26, $vr1
# CHECK-ENCODING: encoding: [0x3a,0x10,0x9e,0x72]

vffinth.d.w $vr18, $vr21
# CHECK-INST: vffinth.d.w $vr18, $vr21
# CHECK-ENCODING: encoding: [0xb2,0x16,0x9e,0x72]

vffint.s.l $vr29, $vr12, $vr7
# CHECK-INST: vffint.s.l $vr29, $vr12, $vr7
# CHECK-ENCODING: encoding: [0x9d,0x1d,0x48,0x71]
