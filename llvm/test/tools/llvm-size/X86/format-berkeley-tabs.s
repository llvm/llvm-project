// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: llvm-size -B -t %t.o | FileCheck %s --strict-whitespace

        .text
        .zero 4
        .data
        .long foo
        .bss
        .zero 4

// Note: this test enables --strict-whitespace to check for literal tabs
// between each field.
// CHECK:      text	{{ *}}data	{{ *}}bss	{{ *}}dec	{{ *}}hex	{{ *}}filename
// CHECK-NEXT: 4	{{ *}}4	{{ *}}4	{{ *}}12	{{ *}}c	{{ *}}{{[ -\(\)_A-Za-z0-9.\\/:]+}}
// CHECK-NEXT: 4	{{ *}}4	{{ *}}4	{{ *}}12	{{ *}}c	{{ *}}(TOTALS)
