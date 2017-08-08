// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=armv7-pc-linux
// RUN: llvm-readobj -t %t.o | FileCheck %s
// RUN: llvm-nm %t.o | FileCheck -allow-empty --check-prefix=NM %s

// Test that nm doesn't print the mapping symbols

// CHECK: Name: $d.0
// NM-NOT: $d.0

        .section        .foobar,"",%progbits
        .asciz  "foo"
        nop
