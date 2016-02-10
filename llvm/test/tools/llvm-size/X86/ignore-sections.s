// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: llvm-size -A %t.o | FileCheck --check-prefix="SYSV" %s
// RUN: llvm-size -B %t.o| FileCheck --check-prefix="BSD" %s

        .text
        .zero 4
        .data
        .long foo
        .bss
        .zero 4
        .ident "foo"
        .section foo
        .long 42
        .cfi_startproc
        .cfi_endproc

// SYSV:    {{[ -\(\)_A-Za-z0-9.\\/:]+}}  :
// SYSV-NEXT:    section             size   addr
// SYSV-NEXT:    .text                  4      0
// SYSV-NEXT:    .data                  4      0
// SYSV-NEXT:    .bss                   4      0
// SYSV-NEXT:    .comment               5      0
// SYSV-NEXT:    foo                    4      0
// SYSV-NEXT:    .eh_frame             48      0
// SYSV-NEXT:    Total                 69

// BSD:        text    data     bss     dec     hex filename
// BSD-NEXT:      4       4       4      12       c {{[ -\(\)_A-Za-z0-9.\\/:]+}}
