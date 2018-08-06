; RUN: opt -licm -basicaa < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' < %s -S | FileCheck %s

; TODO: should be able to hoist both guard and load
define void @test1(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @test1(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 %cond)
; CHECK: %val = load i32, i32* %ptr

entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cond) ["deopt" (i32 0)]
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}

declare void @llvm.experimental.guard(i1, ...)
