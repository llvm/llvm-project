; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-m:e-p:64:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @test_load_cast_combine_tbaa(float* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves TBAA.
; CHECK-LABEL: @test_load_cast_combine_tbaa(
; CHECK: load i32, i32* %{{.*}}, !tbaa !0
entry:
  %l = load float, float* %ptr, !tbaa !0
  %c = bitcast float %l to i32
  ret i32 %c
}

define i32 @test_load_cast_combine_noalias(float* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves no-alias metadata.
; CHECK-LABEL: @test_load_cast_combine_noalias(
; CHECK: load i32, i32* %{{.*}}, !alias.scope !3, !noalias !4
entry:
  %l = load float, float* %ptr, !alias.scope !3, !noalias !4
  %c = bitcast float %l to i32
  ret i32 %c
}

define float @test_load_cast_combine_range(i32* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) drops range metadata. It
; would be nice to preserve or update it somehow but this is hard when moving
; between types.
; CHECK-LABEL: @test_load_cast_combine_range(
; CHECK: load float, float* %{{.*}}
; CHECK-NOT: !range
; CHECK: ret float
entry:
  %l = load i32, i32* %ptr, !range !5
  %c = bitcast i32 %l to float
  ret float %c
}

define i32 @test_load_cast_combine_invariant(float* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves invariant metadata.
; CHECK-LABEL: @test_load_cast_combine_invariant(
; CHECK: load i32, i32* %{{.*}}, !invariant.load !5
entry:
  %l = load float, float* %ptr, !invariant.load !6
  %c = bitcast float %l to i32
  ret i32 %c
}

define i32 @test_load_cast_combine_nontemporal(float* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves nontemporal
; metadata.
; CHECK-LABEL: @test_load_cast_combine_nontemporal(
; CHECK: load i32, i32* %{{.*}}, !nontemporal !6
entry:
  %l = load float, float* %ptr, !nontemporal !7
  %c = bitcast float %l to i32
  ret i32 %c
}

define i8* @test_load_cast_combine_align(i32** %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves align
; metadata.
; CHECK-LABEL: @test_load_cast_combine_align(
; CHECK: load i8*, i8** %{{.*}}, !align !7
entry:
  %l = load i32*, i32** %ptr, !align !8
  %c = bitcast i32* %l to i8*
  ret i8* %c
}

define i8* @test_load_cast_combine_deref(i32** %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves dereferenceable
; metadata.
; CHECK-LABEL: @test_load_cast_combine_deref(
; CHECK: load i8*, i8** %{{.*}}, !dereferenceable !7
entry:
  %l = load i32*, i32** %ptr, !dereferenceable !8
  %c = bitcast i32* %l to i8*
  ret i8* %c
}

define i8* @test_load_cast_combine_deref_or_null(i32** %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves
; dereferenceable_or_null metadata.
; CHECK-LABEL: @test_load_cast_combine_deref_or_null(
; CHECK: load i8*, i8** %{{.*}}, !dereferenceable_or_null !7
entry:
  %l = load i32*, i32** %ptr, !dereferenceable_or_null !8
  %c = bitcast i32* %l to i8*
  ret i8* %c
}

define void @test_load_cast_combine_loop(float* %src, i32* %dst, i32 %n) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves loop access
; metadata.
; CHECK-LABEL: @test_load_cast_combine_loop(
; CHECK: load i32, i32* %{{.*}}, !llvm.mem.parallel_loop_access !4
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %src.gep = getelementptr inbounds float, float* %src, i32 %i
  %dst.gep = getelementptr inbounds i32, i32* %dst, i32 %i
  %l = load float, float* %src.gep, !llvm.mem.parallel_loop_access !4
  %c = bitcast float %l to i32
  store i32 %c, i32* %dst.gep
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit, !llvm.loop !1

exit:
  ret void
}

define void @test_load_cast_combine_nonnull(float** %ptr) {
; We can't preserve nonnull metadata when converting a load of a pointer to
; a load of an integer. Instead, we translate it to range metadata.
; FIXME: We should also transform range metadata back into nonnull metadata.
; FIXME: This test is very fragile. If any LABEL lines are added after
; this point, the test will fail, because this test depends on a metadata tuple,
; which is always emitted at the end of the file. At some point, we should
; consider an option to the IR printer to emit MD tuples after the function
; that first uses them--this will allow us to refer to them like this and not
; have the tests break. For now, this function must always come last in this
; file, and no LABEL lines are to be added after this point.
;
; CHECK-LABEL: @test_load_cast_combine_nonnull(
; CHECK: %[[V:.*]] = load i64, i64* %{{.*}}, !range ![[MD:[0-9]+]]
; CHECK-NOT: !nonnull
; CHECK: store i64 %[[V]], i64*
entry:
  %p = load float*, float** %ptr, !nonnull !6
  %gep = getelementptr float*, float** %ptr, i32 42
  store float* %p, float** %gep
  ret void
}

; This is the metadata tuple that we reference above:
; CHECK: ![[MD]] = !{i64 1, i64 0}
!0 = !{!1, !1, i64 0}
!1 = !{!"scalar type", !2}
!2 = !{!"root"}
!3 = distinct !{!3, !4}
!4 = distinct !{!4}
!5 = !{i32 0, i32 42}
!6 = !{}
!7 = !{i32 1}
!8 = !{i64 8}
