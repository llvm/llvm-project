// RUN: not llvm-mc -triple i386-pc-win32 -filetype=obj < %s 2>&1 |  FileCheck %s

// CHECK: conflicting sections for symbol

        .section .xyz
        .global bar
bar:
        .long 42

        .section        .abcd,"xr",discard,bar
        .global foo
foo:
        .long 42
