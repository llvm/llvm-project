// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux < %s 2>&1 | FileCheck %s

a:
    .section foo
b:
c = b - a

; CHECK: symbol 'a' could not be evaluated in a subtraction expression
