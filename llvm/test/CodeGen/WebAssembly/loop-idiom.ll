; RUN: opt -loop-idiom -S < %s -march=wasm32 | FileCheck %s

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"


; Make sure loop-idiom doesn't create memcpy or memset. These aren't well
; supported in WebAssembly for now.
;
; TODO Check the patterns are recognized once memcpy / memset are supported.

; CHECK-LABEL: @cpy(
; CHECK-NOT: llvm.memcpy
; CHECK: load
; CHECK: store
define void @cpy(i64 %Size) {
bb.nph:
  %Base = alloca i8, i32 10000
  %Dest = alloca i8, i32 10000
  br label %for.body

for.body:
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8, i8* %Base, i64 %indvar
  %DestI = getelementptr i8, i8* %Dest, i64 %indvar
  %V = load i8, i8* %I.0.014, align 1
  store i8 %V, i8* %DestI, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK-LABEL: @set(
; CHECK-NOT: llvm.memset
; CHECK: store
define void @set(i8* %Base, i64 %Size) {
bb.nph:
  br label %for.body

for.body:
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8, i8* %Base, i64 %indvar
  store i8 0, i8* %I.0.014, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
