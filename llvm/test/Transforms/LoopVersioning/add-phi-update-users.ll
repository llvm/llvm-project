; RUN: opt < %s -loop-versioning -S -o - | FileCheck %s

; This test case used to end like this:
;
;    Instruction does not dominate all uses!
;      %t2 = load i16, i16* @b, align 1, !tbaa !2, !alias.scope !6
;      %tobool = icmp eq i16 %t2, 0
;    LLVM ERROR: Broken function found, compilation aborted!
;
; due to a fault where we did not replace the use of %t2 in the icmp in
; for.end, when adding a new PHI node for the versioned loops based on the
; loop-defined values used outside of the loop.
;
; Verify that the code compiles, that we get a versioned loop, and that the
; uses of %t2 in for.end and if.then are updated to use the value from the
; added phi node.

; CHECK:       define void @f1
; CHECK:       for.end.loopexit:
; CHECK-NEXT:    %t2.lver.ph = phi i16 [ %t2.lver.orig, %for.body.lver.orig ]
; CHECK:       for.end.loopexit2:
; CHECK-NEXT:    %t2.lver.ph3 = phi i16 [ %t2, %for.body ]
; CHECK:       for.end:
; CHECK-NEXT:    %t2.lver = phi i16 [ %t2.lver.ph, %for.end.loopexit ], [ %t2.lver.ph3, %for.end.loopexit2 ]
; CHECK-NEXT:    %tobool = icmp eq i16 %t2.lver, 0
; CHECK:       if.then:
; CHECK-NEXT:    store i16 %t2.lver

@a = dso_local global i16 0, align 1
@b = dso_local global i16 0, align 1
@c = dso_local global i16* null, align 1

define void @f1() {
entry:
  %t0 = load i16*, i16** @c, align 1
  br label %for.cond

for.cond:                                         ; preds = %for.cond.backedge, %entry
  br label %for.body

for.body:                                         ; preds = %for.cond, %for.body
  %t1 = phi i64 [ 0, %for.cond ], [ %inc, %for.body ]
  %t2 = load i16, i16* @b, align 1, !tbaa !2
  store i16 %t2, i16* %t0, align 1, !tbaa !2
  %inc = add nuw nsw i64 %t1, 1
  %cmp = icmp ult i64 %inc, 3
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %tobool = icmp eq i16 %t2, 0
  br i1 %tobool, label %for.cond.backedge, label %if.then

for.cond.backedge:                                ; preds = %for.end, %if.then
  br label %for.cond

if.then:                                          ; preds = %for.end
  store i16 %t2, i16* @a, align 1, !tbaa !2
  br label %for.cond.backedge
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 1}
!1 = !{!"clang version 7.0.0"}
!2 = !{!3, !3, i64 0}
!3 = !{!"long long", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
