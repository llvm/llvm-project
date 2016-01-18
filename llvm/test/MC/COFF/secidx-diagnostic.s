// RUN: not llvm-mc -filetype=obj -triple i686-pc-win32 %s 2>%t
// RUN: FileCheck %s < %t

// CHECK: symbol 'bar' can not be undefined

.data
foo:
        .secidx bar
