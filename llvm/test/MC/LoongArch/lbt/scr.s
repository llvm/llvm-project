# RUN: llvm-mc --triple=loongarch64 --show-encoding %s | \
# RUN:        FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s | \
# RUN:        llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INST

movgr2scr $scr0, $a1
# CHECK-INST: movgr2scr $scr0, $a1
# CHECK-ENCODING: encoding: [0xa0,0x08,0x00,0x00]

movscr2gr $a0, $scr1
# CHECK-INST: movscr2gr $a0, $scr1
# CHECK-ENCODING: encoding: [0x24,0x0c,0x00,0x00]

jiscr0 100
# CHECK-INST: jiscr0 100
# CHECK-ENCODING: encoding: [0x00,0x66,0x00,0x48]

jiscr1 100
# CHECK-INST: jiscr1 100
# CHECK-ENCODING: encoding: [0x00,0x67,0x00,0x48]
