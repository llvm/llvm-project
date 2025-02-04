@RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s 2>&1 | FileCheck %s

    .global v
    .text
    movw    r1, #:lower16:v + -20000
    movt    r1, #:upper16:v + 20000

@CHECK-NOT: error: Relocation Not In Range
@CHECK-NOT: movw    r1, #:lower16:v + -20000
@CHECK-NOT: ^
@CHECK-NOT: error: Relocation Not In Range
@CHECK-NOT: movt    r1, #:upper16:v + 20000
@CHECK-NOT: ^
