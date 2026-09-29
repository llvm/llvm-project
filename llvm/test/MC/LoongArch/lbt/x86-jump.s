# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

setx86j $a0, 1
# CHECK-INST: setx86j $a0, 1
# CHECK-ENCODING: encoding: [0x04,0x84,0x36,0x00]

setx86loope $a0, $a1
# CHECK-INST: setx86loope $a0, $a1
# CHECK-ENCODING: encoding: [0xa4,0x78,0x00,0x00]

setx86loopne $a0, $a1
# CHECK-INST: setx86loopne $a0, $a1
# CHECK-ENCODING: encoding: [0xa4,0x7c,0x00,0x00]
