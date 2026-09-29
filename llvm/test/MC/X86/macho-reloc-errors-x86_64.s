// RUN: not llvm-mc -triple=x86_64-apple-darwin -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

        mov %rax, thing
        mov %rax, thing@GOT-thing2@GOT
        mov %rax, (thing-thing2)(%rip)
        mov %rax, thing-thing
        mov %rax, thing-thing2
        mov %rax, thing@PLT
        jmp thing@PLT
        mov %rax, thing@TLVP

// CHECK-ERROR: 3:19: error: 32-bit absolute addressing is not supported in 64-bit mode
// CHECK-ERROR: 4:28: error: expected relocatable expression
// CHECK-ERROR: 5:25: error: unsupported pc-relative relocation of difference
// CHECK-ERROR: 6:24: error: unsupported relocation with identical base
// CHECK-ERROR: 7:24: error: unsupported relocation with subtraction expression, symbol 'thing' can not be undefined in a subtraction expression
// CHECK-ERROR: 8:19: error: unsupported symbol modifier in relocation
// CHECK-ERROR: 9:13: error: unsupported symbol modifier in branch relocation
// CHECK-ERROR: 10:19: error: TLVP symbol modifier should have been rip-rel
