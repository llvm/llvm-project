# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfrecip.s $vr29, $vr14
# CHECK-INST: vfrecip.s $vr29, $vr14
# CHECK-ENCODING: encoding: [0xdd,0xf5,0x9c,0x72]

vfrecip.d $vr24, $vr9
# CHECK-INST: vfrecip.d $vr24, $vr9
# CHECK-ENCODING: encoding: [0x38,0xf9,0x9c,0x72]

vfrecipe.s $vr29, $vr14
# CHECK-INST: vfrecipe.s $vr29, $vr14
# CHECK-ENCODING: encoding: [0xdd,0x15,0x9d,0x72]

vfrecipe.d $vr24, $vr9
# CHECK-INST: vfrecipe.d $vr24, $vr9
# CHECK-ENCODING: encoding: [0x38,0x19,0x9d,0x72]
