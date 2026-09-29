# RUN: not llvm-mc --triple=loongarch32 %s 2>&1 | FileCheck %s --check-prefixes=ERR,ERR32
# RUN: not llvm-mc --triple=loongarch64 %s 2>&1 | FileCheck %s --check-prefix=ERR

## csrxchg: rj != 0,1
csrxchg $a0, $zero, 0
# ERR: :[[#@LINE-1]]:15: error: must not be $r0 or $r1
csrxchg $a0, $ra, 0
# ERR: :[[#@LINE-1]]:15: error: must not be $r0 or $r1

## LoongArch64 mnemonics
iocsrrd.d $a0, $a1
# ERR32: :[[#@LINE-1]]:1: error: instruction requires the following: LA64 Basic Integer and Privilege Instruction Set
iocsrwr.d $a0, $a1
# ERR32: :[[#@LINE-1]]:1: error: instruction requires the following: LA64 Basic Integer and Privilege Instruction Set

## uimm8
lddir $a0, $a0, 0x1ff
# CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [0, 255]
