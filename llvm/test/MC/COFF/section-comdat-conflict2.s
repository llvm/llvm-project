// RUN: not llvm-mc -triple i386-pc-win32 -filetype=obj < %s 2>&1 |  FileCheck %s

// CHECK: <unknown>:0: error: COMDAT symbol 'bar' used by two sections

        .section        .xyz,"xr",discard,bar
        .section        .abcd,"xr",discard,bar
