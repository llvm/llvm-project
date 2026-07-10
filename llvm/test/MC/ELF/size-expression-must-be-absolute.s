// RUN: not --crash llvm-mc -triple=x86_64 -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s

a: b:
.size a, b

// CHECK: LLVM ERROR: size expression must be absolute
