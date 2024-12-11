@RUN: not llvm-mc -triple armv7-eabi -filetype obj -o - %s 2>&1 | FileCheck %s

    .global v
    .text
    movw    r1, #:lower16:v + -65536
    movt    r1, #:upper16:v + 65536

@CHECK: error: Relocation Not In Range
@CHECK: movw    r1, #:lower16:v + -65536
@CHECK: ^
@CHECK: error: Relocation Not In Range
@CHECK: movt    r1, #:upper16:v + 65536
@CHECK: ^
