# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfsqrt.s $vr0, $vr3
# CHECK-INST: vfsqrt.s $vr0, $vr3
# CHECK-ENCODING: encoding: [0x60,0xe4,0x9c,0x72]

vfsqrt.d $vr26, $vr9
# CHECK-INST: vfsqrt.d $vr26, $vr9
# CHECK-ENCODING: encoding: [0x3a,0xe9,0x9c,0x72]
