@RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s 2>&1 | FileCheck %s

    .text
a:
    movw    r1, #:lower16:b - a + 65536
    movt    r1, #:upper16:b - a + 65536
b:

@CHECK-NOT: error: Relocation Not In Range
@CHECK-NOT: movw    r1, #:lower16:b - a + 65536
@CHECK-NOT: ^
@CHECK-NOT: error: Relocation Not In Range
@CHECK-NOT: movt    r1, #:upper16:b - a + 65536
@CHECK-NOT: ^
