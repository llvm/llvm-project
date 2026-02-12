; RUN: opt < %s -mtriple=amdgcn -passes='default<O2>' -S | FileCheck %s
;
; Test type mismatch in ConstantFolding for vector types.

define internal void @f() {
  ret void
}

define void @test() {
; CHECK-LABEL: define void @test(
; CHECK-NEXT:    store <4 x i16> zeroinitializer, ptr @f
; CHECK-NEXT:    ret void
  %p = alloca ptr, addrspace(5)
  %v1 = load <4 x i16>, ptr addrspace(5) %p
  %v2 = load <4 x i16>, ptr addrspace(5) %p
  store ptr @f, ptr addrspace(5) %p
  %sub = sub <4 x i16> %v1, %v2
  %fp = load ptr, ptr addrspace(5) %p
  store <4 x i16> %sub, ptr %fp
  ret void
}
