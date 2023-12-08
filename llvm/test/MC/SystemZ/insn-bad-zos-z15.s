* For z15 only.
* RUN: not llvm-mc -triple s390x-ibm-zos -mcpu=z15 < %s 2> %t
* RUN: FileCheck < %t %s
* RUN: not llvm-mc -triple s390x-ibm-zos -mcpu=arch13 < %s 2> %t
* RUN: FileCheck < %t %s

*CHECK-COUNT-3: error: invalid instruction
        selrnle    0,1,2
        selfhrnle  0,1,2
        selgrnle   0,1,2


*CHECK-COUNT-3: error: invalid instruction
        selrnhe    0,1,2
        selfhrnhe  0,1,2
        selgrnhe   0,1,2


*CHECK-COUNT-3: error: invalid instruction
        selrnlh    0,1,2
        selfhrnlh  0,1,2
        selgrnlh   0,1,2


*CHECK-COUNT-3: error: invalid instruction
        selrlh     0,1,2
        selfhrlh   0,1,2
        selgrlh    0,1,2


*CHECK-COUNT-3: error: invalid instruction
        selrhe     0,1,2
        selfhrhe   0,1,2
        selgrhe    0,1,2


*CHECK-COUNT-3: error: invalid instruction
        selrle     0,1,2
        selfhrle   0,1,2
        selgrle    0,1,2

