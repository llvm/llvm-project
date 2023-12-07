# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

setarmj $a0, 1
# CHECK-INST: setarmj $a0, 1
# CHECK-ENCODING: encoding: [0x04,0xc4,0x36,0x00]
