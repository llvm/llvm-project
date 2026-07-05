# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vpcnt.b $vr2, $vr7
# CHECK-INST: vpcnt.b $vr2, $vr7
# CHECK-ENCODING: encoding: [0xe2,0x20,0x9c,0x72]

vpcnt.h $vr23, $vr25
# CHECK-INST: vpcnt.h $vr23, $vr25
# CHECK-ENCODING: encoding: [0x37,0x27,0x9c,0x72]

vpcnt.w $vr17, $vr24
# CHECK-INST: vpcnt.w $vr17, $vr24
# CHECK-ENCODING: encoding: [0x11,0x2b,0x9c,0x72]

vpcnt.d $vr4, $vr13
# CHECK-INST: vpcnt.d $vr4, $vr13
# CHECK-ENCODING: encoding: [0xa4,0x2d,0x9c,0x72]
