// RUN: not llvm-mc -triple=i686-apple-darwin -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
        .space 0x1000000
        mov %eax, thing-thing2
        mov %eax, defined-thing2
        mov %eax, later-defined

        .section __DATA,__tim
defined:

        .section __DATA,__tim2
later:

// CHECK-ERROR: 3:24: error: symbol 'thing' can not be undefined in a subtraction expression
// CHECK-ERROR: 4:26: error: symbol 'thing2' can not be undefined in a subtraction expression
// CHECK-ERROR: 5:24: error: Section too large, can't encode r_address (0x100000b) into 24 bits of scattered relocation entry.
