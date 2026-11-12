// RUN: llvm-mc %s -filetype=obj -triple=x86_64-pc-linux | llvm-objdump -d - | FileCheck --strict-whitespace --match-full-lines %s
        .text
        .globl  foo
        .type   foo, @function
foo:
        pushq   %rbp
        movq    %rsp, %rbp
        movl    $0, %eax
        popq    %rbp
        ret

        .globl bar
        .type bar, @object
bar:
        .string "test string"

// CHECK:       b: 74 65 73 74 20 73 74 72         test str
// CHECK-NEXT:      13: 69 6e 67 00                     ing.
