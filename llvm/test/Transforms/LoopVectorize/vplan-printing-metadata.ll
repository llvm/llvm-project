; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -disable-output %s 2>&1 | FileCheck %s

define void @test_widen_metadata(ptr noalias %A, ptr noalias %B, i32 %n) {
; CHECK-LABEL: Checking a loop in 'test_widen_metadata'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK:      <x1> vector loop: {
; CHECK:        vector.body:
; CHECK:          WIDEN ir<%lv> = load vp<{{.*}}>
; CHECK:          WIDEN-CAST ir<%conv> = sitofp ir<%lv> to float
; CHECK:          WIDEN ir<%mul> = fmul ir<%conv>, ir<2.000000e+00>
; CHECK:          WIDEN-CAST ir<%conv.back> = fptosi ir<%mul> to i32
; CHECK:          WIDEN store vp<{{.*}}>, ir<%conv.back>
;
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %gep.A = getelementptr inbounds i32, ptr %A, i32 %i
  %lv = load i32, ptr %gep.A, align 4, !tbaa !0, !range !6
  %conv = sitofp i32 %lv to float, !fpmath !5
  %mul = fmul float %conv, 2.0, !fpmath !5
  %conv.back = fptosi float %mul to i32
  %gep.B = getelementptr inbounds i32, ptr %B, i32 %i
  store i32 %conv.back, ptr %gep.B, align 4, !tbaa !0
  %i.next = add i32 %i, 1
  %cond = icmp eq i32 %i.next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

declare float @llvm.sqrt.f32(float)

define void @test_intrinsic_with_metadata(ptr noalias %A, ptr noalias %B, i32 %n) {
; CHECK-LABEL: Checking a loop in 'test_intrinsic_with_metadata'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK:      <x1> vector loop: {
; CHECK:        vector.body:
; CHECK:          WIDEN ir<%lv> = load vp<{{.*}}>
; CHECK:          WIDEN-INTRINSIC ir<%sqrt> = call llvm.sqrt(ir<%lv>)
; CHECK:          WIDEN store vp<{{.*}}>, ir<%sqrt>
;
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %gep.A = getelementptr inbounds float, ptr %A, i32 %i
  %lv = load float, ptr %gep.A, align 4, !tbaa !0
  %sqrt = call float @llvm.sqrt.f32(float %lv), !fpmath !5
  %gep.B = getelementptr inbounds float, ptr %B, i32 %i
  store float %sqrt, ptr %gep.B, align 4, !tbaa !0
  %i.next = add i32 %i, 1
  %cond = icmp eq i32 %i.next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

define void @test_widen_with_multiple_metadata(ptr noalias %A, ptr noalias %B, i32 %n) {
; CHECK-LABEL: Checking a loop in 'test_widen_with_multiple_metadata'
; CHECK:      VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK:      <x1> vector loop: {
; CHECK:        vector.body:
; CHECK:          WIDEN ir<%lv> = load vp<{{.*}}>
; CHECK:          WIDEN-CAST ir<%conv> = sitofp ir<%lv> to float
; CHECK:          WIDEN ir<%mul> = fmul ir<%conv>, ir<2.000000e+00>
; CHECK:          WIDEN-CAST ir<%conv.back> = fptosi ir<%mul> to i32
; CHECK:          WIDEN store vp<{{.*}}>, ir<%conv.back>
;
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %gep.A = getelementptr inbounds i32, ptr %A, i32 %i
  %lv = load i32, ptr %gep.A, align 4, !tbaa !0, !range !6
  %conv = sitofp i32 %lv to float
  %mul = fmul float %conv, 2.0
  %conv.back = fptosi float %mul to i32
  %gep.B = getelementptr inbounds i32, ptr %B, i32 %i
  store i32 %conv.back, ptr %gep.B, align 4, !tbaa !0
  %i.next = add i32 %i, 1
  %cond = icmp eq i32 %i.next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2}
!2 = !{!"root"}
!5 = !{float 2.500000e+00}
!6 = !{i32 0, i32 100}
