// RUN: llvm-mc %s -o %t -filetype=obj -triple=x86_64-pc-linux
// RUN: llvm-nm --print-size %t | FileCheck %s
// RUN: llvm-nm -f sysv %t | FileCheck -check-prefix=SYSV %s

// CHECK: 0000000000000000 ffffffffffffffff n a
// CHECK: 0000000000000000 0000000000000000 N b
// CHECK: 0000000000000004 0000000000000004 C c
// CHECK: ffffffffffffffff 0000000000000000 a d

// SYSV: a                   |0000000000000000|   n  |            NOTYPE|ffffffffffffffff|     |foo
// SYSV: b                   |0000000000000000|   N  |            NOTYPE|0000000000000000|     |foo
// SYSV: c                   |0000000000000004|   C  |            OBJECT|0000000000000004|     |*COM*
// SYSV: d                   |ffffffffffffffff|   a  |            NOTYPE|0000000000000000|     |*ABS*

        .section foo
a:
        .size a, 0xffffffffffffffff

        .global b
b:

        .comm c,4,8

d = 0xffffffffffffffff
