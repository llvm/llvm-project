# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vfnmadd.s $vr26, $vr26, $vr13, $vr9
# CHECK-INST: vfnmadd.s $vr26, $vr26, $vr13, $vr9
# CHECK-ENCODING: encoding: [0x5a,0xb7,0x94,0x09]

vfnmadd.d $vr12, $vr27, $vr31, $vr5
# CHECK-INST: vfnmadd.d $vr12, $vr27, $vr31, $vr5
# CHECK-ENCODING: encoding: [0x6c,0xff,0xa2,0x09]
