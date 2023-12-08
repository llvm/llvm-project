# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vmskgez.b $vr13, $vr0
# CHECK-INST: vmskgez.b $vr13, $vr0
# CHECK-ENCODING: encoding: [0x0d,0x50,0x9c,0x72]
