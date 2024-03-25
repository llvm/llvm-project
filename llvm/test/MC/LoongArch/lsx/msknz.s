# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmsknz.b $vr20, $vr21
# CHECK-INST: vmsknz.b $vr20, $vr21
# CHECK-ENCODING: encoding: [0xb4,0x62,0x9c,0x72]
