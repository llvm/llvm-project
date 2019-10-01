! RUN: llvm-mc %s -arch=sparc  -filetype=obj | llvm-readobj -S --sd | FileCheck %s
! RUN: llvm-mc %s -arch=sparcv9  -filetype=obj | llvm-readobj -S --sd | FileCheck %s

! CHECK: 0000: BA1F401D 01000000 01000000 01000000
! CHECK: 0010: BA1F401D

foo:
        xor %i5, %i5, %i5
        .align 16
        xor %i5, %i5, %i5

