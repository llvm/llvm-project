# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vandi.b $vr10, $vr2, 181
# CHECK-INST: vandi.b $vr10, $vr2, 181
# CHECK-ENCODING: encoding: [0x4a,0xd4,0xd2,0x73]
