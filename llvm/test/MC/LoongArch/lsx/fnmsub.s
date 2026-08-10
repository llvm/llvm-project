# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfnmsub.s $vr2, $vr21, $vr9, $vr2
# CHECK-INST: vfnmsub.s $vr2, $vr21, $vr9, $vr2
# CHECK-ENCODING: encoding: [0xa2,0x26,0xd1,0x09]

vfnmsub.d $vr4, $vr12, $vr27, $vr19
# CHECK-INST: vfnmsub.d $vr4, $vr12, $vr27, $vr19
# CHECK-ENCODING: encoding: [0x84,0xed,0xe9,0x09]
