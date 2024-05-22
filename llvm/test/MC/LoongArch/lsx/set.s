# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

vseteqz.v $fcc0, $vr13
# CHECK-INST: vseteqz.v $fcc0, $vr13
# CHECK-ENCODING: encoding: [0xa0,0x99,0x9c,0x72]

vsetnez.v $fcc7, $vr14
# CHECK-INST: vsetnez.v $fcc7, $vr14
# CHECK-ENCODING: encoding: [0xc7,0x9d,0x9c,0x72]
