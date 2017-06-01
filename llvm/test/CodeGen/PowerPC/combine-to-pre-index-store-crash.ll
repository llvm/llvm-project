; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; CHECK-LABEL: TestFoo:
; CHECK: std
; CHECK: bl TestBar
; CHECK: stbu
; CHECK: std
; CHECK: blr

%StructA = type <{ i64, { i64, i64 }, { i64, i64 } }>

define void @TestFoo(%StructA* %this) {
  %tmp = getelementptr inbounds %StructA, %StructA* %this, i64 0, i32 1
  %tmp11 = getelementptr inbounds %StructA, %StructA* %this, i64 0, i32 1, i32 1
  %tmp12 = bitcast { i64, i64 }* %tmp to i64**
  store i64* %tmp11, i64** %tmp12
  call void @TestBar()
  %tmp13 = getelementptr inbounds %StructA, %StructA* %this, i64 0, i32 2, i32 1
  store i64* %tmp13, i64** undef
  %.cast.i.i.i = bitcast i64* %tmp13 to i8*
  store i8 0, i8* %.cast.i.i.i
  ret void
}

declare void @TestBar()
