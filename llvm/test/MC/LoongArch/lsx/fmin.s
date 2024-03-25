# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfmin.s $vr18, $vr17, $vr1
# CHECK-INST: vfmin.s $vr18, $vr17, $vr1
# CHECK-ENCODING: encoding: [0x32,0x86,0x3e,0x71]

vfmin.d $vr30, $vr12, $vr5
# CHECK-INST: vfmin.d $vr30, $vr12, $vr5
# CHECK-ENCODING: encoding: [0x9e,0x15,0x3f,0x71]
