# RUN: llvm-mc -triple i386-unknown-unknown %s -I  %p | FileCheck %s

# CHECK: .byte 2
.if 1+2
    .if 1-1
        .byte 1
    .elseif 2+2
        .byte 1+1
    .else
        .byte 0
    .endif
.endif
