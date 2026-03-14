# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vneg.b $vr11, $vr29
# CHECK-INST: vneg.b $vr11, $vr29
# CHECK-ENCODING: encoding: [0xab,0x33,0x9c,0x72]

vneg.h $vr14, $vr4
# CHECK-INST: vneg.h $vr14, $vr4
# CHECK-ENCODING: encoding: [0x8e,0x34,0x9c,0x72]

vneg.w $vr4, $vr0
# CHECK-INST: vneg.w $vr4, $vr0
# CHECK-ENCODING: encoding: [0x04,0x38,0x9c,0x72]

vneg.d $vr0, $vr5
# CHECK-INST: vneg.d $vr0, $vr5
# CHECK-ENCODING: encoding: [0xa0,0x3c,0x9c,0x72]
