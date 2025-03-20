; RUN: not --crash llc < %s -mtriple=nvptx -mcpu=sm_20 2>&1 | FileCheck %s

; Error out if initializer is given for address spaces that do not support initializers
; CHECK: LLVM ERROR: initial value of 'g0' is not allowed in addrspace(3)
@g0 = addrspace(3) global i32 42
