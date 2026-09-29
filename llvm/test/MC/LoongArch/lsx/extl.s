# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vextl.q.d $vr14, $vr20
# CHECK-INST: vextl.q.d $vr14, $vr20
# CHECK-ENCODING: encoding: [0x8e,0x02,0x09,0x73]

vextl.qu.du $vr26, $vr26
# CHECK-INST: vextl.qu.du $vr26, $vr26
# CHECK-ENCODING: encoding: [0x5a,0x03,0x0d,0x73]
