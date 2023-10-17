# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfmsub.s $vr25, $vr30, $vr4, $vr13
# CHECK-INST: vfmsub.s $vr25, $vr30, $vr4, $vr13
# CHECK-ENCODING: encoding: [0xd9,0x93,0x56,0x09]

vfmsub.d $vr3, $vr1, $vr0, $vr19
# CHECK-INST: vfmsub.d $vr3, $vr1, $vr0, $vr19
# CHECK-ENCODING: encoding: [0x23,0x80,0x69,0x09]
