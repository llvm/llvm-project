# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vsle.b $vr4, $vr30, $vr18
# CHECK-INST: vsle.b $vr4, $vr30, $vr18
# CHECK-ENCODING: encoding: [0xc4,0x4b,0x02,0x70]

vsle.h $vr3, $vr13, $vr12
# CHECK-INST: vsle.h $vr3, $vr13, $vr12
# CHECK-ENCODING: encoding: [0xa3,0xb1,0x02,0x70]

vsle.w $vr21, $vr17, $vr20
# CHECK-INST: vsle.w $vr21, $vr17, $vr20
# CHECK-ENCODING: encoding: [0x35,0x52,0x03,0x70]

vsle.d $vr22, $vr0, $vr28
# CHECK-INST: vsle.d $vr22, $vr0, $vr28
# CHECK-ENCODING: encoding: [0x16,0xf0,0x03,0x70]

vslei.b $vr8, $vr11, 4
# CHECK-INST: vslei.b $vr8, $vr11, 4
# CHECK-ENCODING: encoding: [0x68,0x11,0x82,0x72]

vslei.h $vr15, $vr22, 0
# CHECK-INST: vslei.h $vr15, $vr22, 0
# CHECK-ENCODING: encoding: [0xcf,0x82,0x82,0x72]

vslei.w $vr23, $vr17, 12
# CHECK-INST: vslei.w $vr23, $vr17, 12
# CHECK-ENCODING: encoding: [0x37,0x32,0x83,0x72]

vslei.d $vr11, $vr18, -12
# CHECK-INST: vslei.d $vr11, $vr18, -12
# CHECK-ENCODING: encoding: [0x4b,0xd2,0x83,0x72]

vsle.bu $vr20, $vr11, $vr31
# CHECK-INST: vsle.bu $vr20, $vr11, $vr31
# CHECK-ENCODING: encoding: [0x74,0x7d,0x04,0x70]

vsle.hu $vr5, $vr6, $vr7
# CHECK-INST: vsle.hu $vr5, $vr6, $vr7
# CHECK-ENCODING: encoding: [0xc5,0x9c,0x04,0x70]

vsle.wu $vr15, $vr14, $vr22
# CHECK-INST: vsle.wu $vr15, $vr14, $vr22
# CHECK-ENCODING: encoding: [0xcf,0x59,0x05,0x70]

vsle.du $vr0, $vr29, $vr17
# CHECK-INST: vsle.du $vr0, $vr29, $vr17
# CHECK-ENCODING: encoding: [0xa0,0xc7,0x05,0x70]

vslei.bu $vr12, $vr27, 12
# CHECK-INST: vslei.bu $vr12, $vr27, 12
# CHECK-ENCODING: encoding: [0x6c,0x33,0x84,0x72]

vslei.hu $vr22, $vr31, 12
# CHECK-INST: vslei.hu $vr22, $vr31, 12
# CHECK-ENCODING: encoding: [0xf6,0xb3,0x84,0x72]

vslei.wu $vr19, $vr18, 21
# CHECK-INST: vslei.wu $vr19, $vr18, 21
# CHECK-ENCODING: encoding: [0x53,0x56,0x85,0x72]

vslei.du $vr19, $vr14, 26
# CHECK-INST: vslei.du $vr19, $vr14, 26
# CHECK-ENCODING: encoding: [0xd3,0xe9,0x85,0x72]
