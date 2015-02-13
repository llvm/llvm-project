; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-m:e-p:64:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @test_load_cast_combine_tbaa(float* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves TBAA.
; CHECK-LABEL: @test_load_cast_combine_tbaa(
; CHECK: load i32* %{{.*}}, !tbaa !0
entry:
  %l = load float* %ptr, !tbaa !0
  %c = bitcast float %l to i32
  ret i32 %c
}

define i32 @test_load_cast_combine_noalias(float* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves no-alias metadata.
; CHECK-LABEL: @test_load_cast_combine_noalias(
; CHECK: load i32* %{{.*}}, !alias.scope !2, !noalias !1
entry:
  %l = load float* %ptr, !alias.scope !2, !noalias !1
  %c = bitcast float %l to i32
  ret i32 %c
}

define float @test_load_cast_combine_range(i32* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) drops range metadata. It
; would be nice to preserve or update it somehow but this is hard when moving
; between types.
; CHECK-LABEL: @test_load_cast_combine_range(
; CHECK: load float* %{{.*}}
; CHECK-NOT: !range
; CHECK: ret float
entry:
  %l = load i32* %ptr, !range !5
  %c = bitcast i32 %l to float
  ret float %c
}

define i32 @test_load_cast_combine_invariant(float* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves invariant metadata.
; CHECK-LABEL: @test_load_cast_combine_invariant(
; CHECK: load i32* %{{.*}}, !invariant.load !3
entry:
  %l = load float* %ptr, !invariant.load !3
  %c = bitcast float %l to i32
  ret i32 %c
}

define i32 @test_load_cast_combine_nontemporal(float* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves nontemporal
; metadata.
; CHECK-LABEL: @test_load_cast_combine_nontemporal(
; CHECK: load i32* %{{.*}}, !nontemporal !4
entry:
  %l = load float* %ptr, !nontemporal !4
  %c = bitcast float %l to i32
  ret i32 %c
}

define void @test_load_cast_combine_loop(float* %src, i32* %dst, i32 %n) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves loop access
; metadata.
; CHECK-LABEL: @test_load_cast_combine_loop(
; CHECK: load i32* %{{.*}}, !llvm.mem.parallel_loop_access !1
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %src.gep = getelementptr inbounds float* %src, i32 %i
  %dst.gep = getelementptr inbounds i32* %dst, i32 %i
  %l = load float* %src.gep, !llvm.mem.parallel_loop_access !1
  %c = bitcast float %l to i32
  store i32 %c, i32* %dst.gep
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit, !llvm.loop !1

exit:
  ret void
}

!0 = !{ !1, !1, i64 0 }
!1 = !{ !1 }
!2 = !{ !2, !1 }
!3 = !{ }
!4 = !{ i32 1 }
!5 = !{ i32 0, i32 42 }
