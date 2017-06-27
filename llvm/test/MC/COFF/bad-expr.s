// RUN: not llvm-mc -filetype=obj -triple i386-pc-win32 %s 2>&1 | FileCheck %s

// CHECK: symbol '__ImageBase' can not be undefined in a subtraction expression

        .data
_x:
        .long   _x-__ImageBase
