// RUN: not --crash llvm-mc -triple=x86_64-apple-darwin -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

.align 4
.data_region jt32
foo:
     .long 0

// CHECK-ERROR: LLVM ERROR: Data region not terminated
