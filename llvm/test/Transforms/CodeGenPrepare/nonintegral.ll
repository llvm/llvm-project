; RUN: opt -S -codegenprepare < %s | FileCheck %s
; RUN: opt -S -codegenprepare -addr-sink-using-gep=false < %s | FileCheck %s

; This target data layout is modified to have a non-integral addrspace(1),
; in order to verify that codegenprepare does not try to introduce illegal
; inttoptrs.
target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

define void @test_simple(i1 %cond, i64 addrspace(1)* %base) {
; CHECK-LABEL: @test_simple
; CHECK-NOT: inttoptr {{.*}} to i64 addrspace(1)*
entry:
  %addr = getelementptr inbounds i64, i64 addrspace(1)* %base, i64 5
  %casted = bitcast i64 addrspace(1)* %addr to i32 addrspace(1)*
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %v = load i32, i32 addrspace(1)* %casted, align 4
  br label %fallthrough

fallthrough:
  ret void
}


define void @test_inttoptr_base(i1 %cond, i64 %base) {
; CHECK-LABEL: @test_inttoptr_base
; CHECK-NOT: inttoptr {{.*}} to i64 addrspace(1)*
entry:
; Doing the inttoptr in the integral addrspace(0) followed by an explicit
; (frontend-introduced) addrspacecast is fine. We cannot however introduce
; a direct inttoptr to addrspace(1)
  %baseptr = inttoptr i64 %base to i64*
  %baseptrni = addrspacecast i64 *%baseptr to i64 addrspace(1)*
  %addr = getelementptr inbounds i64, i64 addrspace(1)* %baseptrni, i64 5
  %casted = bitcast i64 addrspace(1)* %addr to i32 addrspace(1)*
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %v = load i32, i32 addrspace(1)* %casted, align 4
  br label %fallthrough

fallthrough:
  ret void
}

define void @test_ptrtoint_base(i1 %cond, i64 addrspace(1)* %base) {
; CHECK-LABEL: @test_ptrtoint_base
; CHECK-NOT: ptrtoint addrspace(1)* {{.*}} to i64
entry:
; This one is inserted by the frontend, so it's fine. We're not allowed to
; directly ptrtoint %base ourselves though
  %baseptr0 = addrspacecast i64 addrspace(1)* %base to i64*
  %toint = ptrtoint i64* %baseptr0 to i64
  %added = add i64 %toint, 8
  %toptr = inttoptr i64 %added to i64*
  %geped = getelementptr i64, i64* %toptr, i64 2
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %v = load i64, i64* %geped, align 4
  br label %fallthrough

fallthrough:
  ret void
}
