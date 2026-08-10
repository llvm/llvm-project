# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vxor.v $vr28, $vr16, $vr18
# CHECK-INST: vxor.v $vr28, $vr16, $vr18
# CHECK-ENCODING: encoding: [0x1c,0x4a,0x27,0x71]
