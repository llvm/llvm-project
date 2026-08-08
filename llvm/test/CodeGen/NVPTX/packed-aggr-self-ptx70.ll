; RUN: not --crash llc < %s -mtriple=nvptx -mcpu=sm_20 -mattr=+ptx70 2>&1 | FileCheck %s

; Self-referential packed aggregates still require mask(), which is available
; only in PTX ISA 7.1 and later.

; CHECK: LLVM ERROR: initialized packed aggregate with pointers 'self_packed' requires at least PTX ISA version 7.1

%t = type <{ ptr, i8 }>
@self_packed = addrspace(1) global %t <{
  ptr addrspacecast (ptr addrspace(1) getelementptr (i8, ptr addrspace(1) @self_packed, i32 3) to ptr),
  i8 7 }>, align 1
