; RUN: opt -S -passes=loop-rotate -verify-memoryssa < %s | FileCheck %s
; RUN: opt -S -passes='require<target-ir>,require<assumptions>,loop(loop-rotate)' < %s | FileCheck %s
; RUN: opt -S -passes='require<target-ir>,require<assumptions>,loop(loop-rotate)' -verify-memoryssa  < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(ptr)

define void @test_02(ptr nocapture %_pA) nounwind ssp {
; CHECK-LABEL: @test_02(
; CHECK: entry:
; CHECK:   tail call void @llvm.experimental.noalias.scope.decl(metadata !2)
; CHECK:   store i32 42, ptr %_pA, align 16, !alias.scope !2
; CHECK: for.body:
; CHECK:   tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
; CHECK:   store i32 0, ptr %array, align 16, !noalias !5
; CHECK:   tail call void @llvm.experimental.noalias.scope.decl(metadata !7)
; CHECK:   store i32 42, ptr %_pA, align 16, !alias.scope !7
; CHECK: for.end:

entry:
  %array = alloca [20 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2)
  store i32 42, ptr %_pA, align 16, !alias.scope !2
  %cmp = icmp slt i32 %i.0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %array, align 16, !noalias !2
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %arrayidx.lcssa = phi ptr [ %array, %for.cond ]
  call void @g(ptr %arrayidx.lcssa) nounwind
  ret void
}

define void @test_03(ptr nocapture %_pA) nounwind ssp {
; CHECK-LABEL: @test_03(
; CHECK: entry:
; CHECK: for.body:
; CHECK:   tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
; CHECK:   store i32 42, ptr %_pA, align 16, !alias.scope !5
; CHECK:   store i32 0, ptr %array, align 16, !noalias !5
; CHECK: for.end:

entry:
  %array = alloca [20 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2)
  store i32 42, ptr %_pA, align 16, !alias.scope !2
  store i32 0, ptr %array, align 16, !noalias !2
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %arrayidx.lcssa = phi ptr [ %array, %for.cond ]
  call void @g(ptr %arrayidx.lcssa) nounwind
  ret void
}

define void @test_04(ptr nocapture %_pA) nounwind ssp {
; CHECK-LABEL: @test_04(
; CHECK: entry:
; CHECK:   tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
; CHECK:   store i32 42, ptr %_pA, align 16, !alias.scope !9
; CHECK: for.body:
; CHECK:   tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
; CHECK:   store i32 0, ptr %array, align 16, !noalias !5
; CHECK:   store i32 43, ptr %_pA, align 16, !alias.scope !5
; CHECK:   tail call void @llvm.experimental.noalias.scope.decl(metadata !11)
; CHECK:   store i32 42, ptr %_pA, align 16, !alias.scope !11
; CHECK: for.end:
entry:
  %array = alloca [20 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2)
  store i32 42, ptr %_pA, align 16, !alias.scope !2
  %cmp = icmp slt i32 %i.0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %array, align 16, !noalias !2
  store i32 43, ptr %_pA, align 16, !alias.scope !2
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %arrayidx.lcssa = phi ptr [ %array, %for.cond ]
  call void @g(ptr %arrayidx.lcssa) nounwind
  ret void
}

define void @test_05(ptr nocapture %_pA) nounwind ssp {
; CHECK-LABEL: @test_05(
; CHECK: entry:
; CHECK:   tail call void @llvm.experimental.noalias.scope.decl(metadata !13)
; CHECK:   store i32 42, ptr %_pA, align 16, !alias.scope !13
; CHECK: for.body:
; CHECK:   tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
; CHECK:   store i32 0, ptr %array, align 16, !noalias !5
; CHECK:   store i32 43, ptr %_pA, align 16, !alias.scope !5
; CHECK:   tail call void @llvm.experimental.noalias.scope.decl(metadata !15)
; CHECK:   store i32 42, ptr %_pA, align 16, !alias.scope !15
; CHECK: for.end:
; CHECK:   store i32 44, ptr %_pA, align 16, !alias.scope !5

entry:
  %array = alloca [20 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2)
  store i32 42, ptr %_pA, align 16, !alias.scope !2
  %cmp = icmp slt i32 %i.0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %array, align 16, !noalias !2
  store i32 43, ptr %_pA, align 16, !alias.scope !2
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %arrayidx.lcssa = phi ptr [ %array, %for.cond ]
  store i32 44, ptr %_pA, align 16, !alias.scope !2
  call void @g(ptr %arrayidx.lcssa) nounwind
  ret void
}

; Function Attrs: inaccessiblememonly nounwind
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inaccessiblememonly nounwind }
attributes #2 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = !{!3}
!3 = distinct !{!3, !4, !"test_loop_rotate_XX: pA"}
!4 = distinct !{!4, !"test_loop_rotate_XX"}

; CHECK: !0 = !{i32 1, !"wchar_size", i32 4}
; CHECK: !1 = !{!"clang"}
; CHECK: !2 = !{!3}
; CHECK: !3 = distinct !{!3, !4, !"test_loop_rotate_XX: pA:pre.rot"}
; CHECK: !4 = distinct !{!4, !"test_loop_rotate_XX"}
; CHECK: !5 = !{!6}
; CHECK: !6 = distinct !{!6, !4, !"test_loop_rotate_XX: pA"}
; CHECK: !7 = !{!8}
; CHECK: !8 = distinct !{!8, !4, !"test_loop_rotate_XX: pA:h.rot"}
; CHECK: !9 = !{!10}
; CHECK: !10 = distinct !{!10, !4, !"test_loop_rotate_XX: pA:pre.rot"}
; CHECK: !11 = !{!12}
; CHECK: !12 = distinct !{!12, !4, !"test_loop_rotate_XX: pA:h.rot"}
; CHECK: !13 = !{!14}
; CHECK: !14 = distinct !{!14, !4, !"test_loop_rotate_XX: pA:pre.rot"}
; CHECK: !15 = !{!16}
; CHECK: !16 = distinct !{!16, !4, !"test_loop_rotate_XX: pA:h.rot"}
