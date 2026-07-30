# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

armmove $a0, $a1, 1
# CHECK-INST: armmove $a0, $a1, 1
# CHECK-ENCODING: encoding: [0xa4,0x44,0x36,0x00]

armmov.w $a0, 1
# CHECK-INST: armmov.w $a0, 1
# CHECK-ENCODING: encoding: [0x9d,0xc4,0x3f,0x00]

armmov.d $a0, 1
# CHECK-INST: armmov.d $a0, 1
# CHECK-ENCODING: encoding: [0x9e,0xc4,0x3f,0x00]

armmfflag $a0, 1
# CHECK-INST: armmfflag $a0, 1
# CHECK-ENCODING: encoding: [0x44,0x04,0x5c,0x00]

armmtflag $a0, 1
# CHECK-INST: armmtflag $a0, 1
# CHECK-ENCODING: encoding: [0x64,0x04,0x5c,0x00]
