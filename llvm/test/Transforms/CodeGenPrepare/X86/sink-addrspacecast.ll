; RUN: opt -S -codegenprepare < %s | FileCheck %s -check-prefix=CHECK -check-prefix=GEP

target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @load_cast_gep
; GEP: [[CAST:%[0-9]+]] = addrspacecast ptr %base to ptr addrspace(1)
; GEP: getelementptr inbounds i8, ptr addrspace(1) [[CAST]], i64 40
define void @load_cast_gep(i1 %cond, ptr %base) {
entry:
  %addr = getelementptr inbounds i64, ptr %base, i64 5
  %casted = addrspacecast ptr %addr to ptr addrspace(1)
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %v = load i32, ptr addrspace(1) %casted, align 4
  br label %fallthrough

fallthrough:
  ret void
}

; CHECK-LABEL: @store_gep_cast
; GEP: [[CAST:%[0-9]+]] = addrspacecast ptr %base to ptr addrspace(1)
; GEP: getelementptr inbounds i8, ptr addrspace(1) [[CAST]], i64 20
define void @store_gep_cast(i1 %cond, ptr %base) {
entry:
  %casted = addrspacecast ptr %base to ptr addrspace(1)
  %addr = getelementptr inbounds i32, ptr addrspace(1) %casted, i64 5
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  store i32 0, ptr addrspace(1) %addr, align 4
  br label %fallthrough

fallthrough:
  ret void
}
