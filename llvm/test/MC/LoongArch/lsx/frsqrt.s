# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfrsqrt.s $vr19, $vr30
# CHECK-INST: vfrsqrt.s $vr19, $vr30
# CHECK-ENCODING: encoding: [0xd3,0x07,0x9d,0x72]

vfrsqrt.d $vr1, $vr0
# CHECK-INST: vfrsqrt.d $vr1, $vr0
# CHECK-ENCODING: encoding: [0x01,0x08,0x9d,0x72]
