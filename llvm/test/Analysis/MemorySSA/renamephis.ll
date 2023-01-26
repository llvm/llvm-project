; RUN: opt -passes=licm -verify-memoryssa -S %s | FileCheck %s
; REQUIRES: asserts
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@0 = external global { { ptr, i32, i32 }, ptr, i8, i8 }

declare void @g()

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #0

; CHECK-LABEL: @f
define void @f() align 2 {
entry:
  %P = alloca ptr, align 8
  br label %cond.end.i.i.i.i

cond.end.i.i.i.i:                                 ; preds = %cont20, %entry
  br i1 undef, label %cont20, label %if.end

cont20:                                           ; preds = %cond.end.i.i.i.i, %cond.end.i.i.i.i, %cond.end.i.i.i.i
  store ptr undef, ptr %P, align 8
  br label %cond.end.i.i.i.i

if.end:                                           ; preds = %cond.end.i.i.i.i
  br i1 undef, label %cond.exit, label %handler.type_mismatch2.i

handler.type_mismatch2.i:                         ; preds = %if.end
  tail call void @g()
  unreachable

cond.exit:             ; preds = %if.end
  switch i8 undef, label %block.exit [
    i8 81, label %sw.bb94
    i8 12, label %cleanup
    i8 74, label %cleanup
  ]

block.exit: ; preds = %cond.exit
  unreachable

sw.bb94:                                          ; preds = %cond.exit
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull undef)
  br label %cleanup

cleanup:                                          ; preds = %sw.bb94, %cond.exit, %cond.exit
  ret void
}

attributes #0 = { argmemonly nounwind willreturn }
