; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; CHECK-LABEL: TestFoo:
; CHECK: std
; CHECK: bl TestBar
; CHECK: stbu
; CHECK: std
; CHECK: blr

%StructA = type <{ i64, { i64, i64 }, { i64, i64 } }>

define void @TestFoo(ptr %this) {
  %tmp = getelementptr inbounds %StructA, ptr %this, i64 0, i32 1
  %tmp11 = getelementptr inbounds %StructA, ptr %this, i64 0, i32 1, i32 1
  store ptr %tmp11, ptr %tmp
  call void @TestBar()
  %tmp13 = getelementptr inbounds %StructA, ptr %this, i64 0, i32 2, i32 1
  store ptr %tmp13, ptr undef
  store i8 0, ptr %tmp13
  ret void
}

declare void @TestBar()
