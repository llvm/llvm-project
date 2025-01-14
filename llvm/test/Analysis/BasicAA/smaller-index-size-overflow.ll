; RUN: opt -S -passes=aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "p1:32:32"

; CHECK: PartialAlias:	i32 addrspace(1)* %gep1, i32 addrspace(1)* %gep2
define void @test(ptr addrspace(1) %p) {
  %gep1 = getelementptr i8, ptr addrspace(1) %p, i32 u0x7fffffff
  %gep2 = getelementptr i8, ptr addrspace(1) %p, i32 u0x80000001
  store i32 0, ptr addrspace(1) %gep1
  load i32, ptr addrspace(1) %gep2
  ret void
}
