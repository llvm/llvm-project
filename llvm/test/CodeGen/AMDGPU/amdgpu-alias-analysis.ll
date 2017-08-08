; RUN: opt -mtriple=amdgcn-- -O3 -aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=r600-- -O3 -aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

; CHECK: NoAlias:      i8 addrspace(1)* %p1, i8* %p

define void @test(i8* %p, i8 addrspace(1)* %p1) {
  ret void
}

