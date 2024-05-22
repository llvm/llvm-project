# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfmul.s $vr16, $vr8, $vr17
# CHECK-INST: vfmul.s $vr16, $vr8, $vr17
# CHECK-ENCODING: encoding: [0x10,0xc5,0x38,0x71]

vfmul.d $vr3, $vr6, $vr1
# CHECK-INST: vfmul.d $vr3, $vr6, $vr1
# CHECK-ENCODING: encoding: [0xc3,0x04,0x39,0x71]
