@RUN: llvm-mc -triple armv7-eabi -filetype obj %s -o - | llvm-objdump -d --triple armv7-eabi - | FileCheck %s

a:
    movw    r1, #:lower16:b - a + 65536
    movt    r1, #:upper16:b - a + 65536
b:

@CHECK: 0: e3001008 movw r1, #0x8
@CHECK: 4: e3401001 movt r1, #0x1
