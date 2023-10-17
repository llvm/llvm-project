# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vshuf.b $vr27, $vr17, $vr31, $vr28
# CHECK-INST: vshuf.b $vr27, $vr17, $vr31, $vr28
# CHECK-ENCODING: encoding: [0x3b,0x7e,0x5e,0x0d]

vshuf.h $vr21, $vr10, $vr31
# CHECK-INST: vshuf.h $vr21, $vr10, $vr31
# CHECK-ENCODING: encoding: [0x55,0xfd,0x7a,0x71]

vshuf.w $vr18, $vr17, $vr23
# CHECK-INST: vshuf.w $vr18, $vr17, $vr23
# CHECK-ENCODING: encoding: [0x32,0x5e,0x7b,0x71]

vshuf.d $vr4, $vr24, $vr11
# CHECK-INST: vshuf.d $vr4, $vr24, $vr11
# CHECK-ENCODING: encoding: [0x04,0xaf,0x7b,0x71]
