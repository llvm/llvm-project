;; This test ensures that the logic which assigns calls to stack nodes
;; correctly handles an inlined callsite with stack ids that partially
;; overlap with a trimmed context. In particular when it also partially
;; overlaps with a longer non-trimmed context that doesn't match all of
;; the inlined callsite stack ids.

;; The profile data and call stacks were all manually added, but the code
;; would be structured something like the following (fairly contrived to
;; result in the type of control flow needed to test):

;; void A(bool b) {
;;   if (b)
;;     // cold: stack ids 6, 2, 8 (trimmed ids 10)
;;     // not cold: stack ids 6, 7 (trimmed ids 9, 11)
;;     new char[10]; // stack id 6
;;   else
;;     // not cold: stack ids 1, 2, 8, 3, 4
;;     // cold: stack ids 1, 2, 8, 3, 5
;;     new char[10]; // stack id 1
;; }
;;
;; void XZ() {
;;   A(false); // stack ids 2, 8 (e.g. X inlined into Z)
;; }
;;
;; void XZN() {
;;   // This is the tricky one to get right. We want to ensure it gets
;;   // correctly correlated with a stack node for the trimmed 6, 2, 8
;;   // context shown in A. It should *not* be correlated with the longer
;;   // untrimmed 1, 2, 8, 3, 4|5 contexts.
;;   A(true); // stack ids 2, 8, 9 (e.g. X inlined into Z inlined into N)
;; }
;;
;; void Y() {
;;   A(true); // stack id 7
;; }
;;
;; void M() {
;;   XZ(); // stack id 3
;; }
;;
;; int main() {
;;   M(); // stack id 4 (leads to not cold allocation)
;;   M(); // stack id 5 (leads to cold allocation)
;;   XZN(); // stack id 11 (leads to cold allocation)
;;   Y(); // stack id 10 (leads to not cold allocation)
;; }

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=IR \
; RUN:  --check-prefix=STATS --check-prefix=REMARKS

; REMARKS: created clone _Z1Ab.memprof.1
; REMARKS: created clone _Z2XZv.memprof.1
; REMARKS: created clone _Z1Mv.memprof.1
;; Make sure the inlined context in _Z3XZNv, which partially overlaps
;; trimmed cold context, and also partially overlaps completely
;; unrelated contexts, correctly calls a cloned version of Z1Ab,
;; which will call the cold annotated allocation.
; REMARKS: call in clone _Z3XZNv assigned to call function clone _Z1Ab.memprof.1
; REMARKS: call in clone main assigned to call function clone _Z1Mv.memprof.1
; REMARKS: call in clone _Z1Mv.memprof.1 assigned to call function clone _Z2XZv.memprof.1
; REMARKS: call in clone _Z2XZv.memprof.1 assigned to call function clone _Z1Ab
; REMARKS: call in clone main assigned to call function clone _Z1Mv
; REMARKS: call in clone _Z1Mv assigned to call function clone _Z2XZv
; REMARKS: call in clone _Z2XZv assigned to call function clone _Z1Ab.memprof.1
; REMARKS: call in clone _Z1Ab.memprof.1 marked with memprof allocation attribute cold
; REMARKS: call in clone _Z1Yv assigned to call function clone _Z1Ab
; REMARKS: call in clone _Z1Ab marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z1Ab marked with memprof allocation attribute cold
; REMARKS: call in clone _Z1Ab.memprof.1 marked with memprof allocation attribute notcold


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z1Ab(i1 noundef zeroext %b) {
entry:
  br i1 %b, label %if.then, label %if.else

if.then:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7, !memprof !5, !callsite !11
  br label %if.end

if.else:
  %call2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7, !memprof !0, !callsite !10
  br label %if.end

if.end:
  ret void
}

; Function Attrs: nobuiltin
declare ptr @_Znam(i64) #0

define dso_local void @_Z2XZv() local_unnamed_addr #0 {
entry:
  tail call void @_Z1Ab(i1 noundef zeroext false), !callsite !12
  ret void
}

define dso_local void @_Z1Mv() local_unnamed_addr #0 {
entry:
  tail call void @_Z2XZv(), !callsite !19
  ret void
}

define dso_local void @_Z3XZNv() local_unnamed_addr {
entry:
  tail call void @_Z1Ab(i1 noundef zeroext true), !callsite !15
  ret void
}

define dso_local void @_Z1Yv() local_unnamed_addr {
entry:
  tail call void @_Z1Ab(i1 noundef zeroext true), !callsite !17
  ret void
}

define dso_local noundef i32 @main() local_unnamed_addr {
entry:
  tail call void @_Z1Mv(), !callsite !13 ;; Not cold context
  tail call void @_Z1Mv(), !callsite !14 ;; Cold context
  tail call void @_Z3XZNv(), !callsite !16 ;; Cold context
  tail call void @_Z1Yv(), !callsite !18 ;; Not cold context
  ret i32 0
}

attributes #0 = { nobuiltin }
attributes #7 = { builtin }

!0 = !{!1, !3}
;; Not cold context via first call to _Z1Mv in main
!1 = !{!2, !"notcold"}
!2 = !{i64 1, i64 2, i64 8, i64 3, i64 4}
;; Cold context via second call to _Z1Mv in main
!3 = !{!4, !"cold"}
!4 = !{i64 1, i64 2, i64 8, i64 3, i64 5}
!5 = !{!6, !8}
;; Cold (trimmed) context via call to _Z3XZNv in main
!6 = !{!7, !"cold"}
!7 = !{i64 6, i64 2, i64 8}
;; Not cold (trimmed) context via call to _Z1Yv in main
!8 = !{!9, !"notcold"}
!9 = !{i64 6, i64 7}
!10 = !{i64 1}
!11 = !{i64 6}
!12 = !{i64 2, i64 8}
!13 = !{i64 4}
!14 = !{i64 5}
;; Inlined context in _Z3XZNv, which includes part of trimmed cold context
!15 = !{i64 2, i64 8, i64 9}
!16 = !{i64 11}
!17 = !{i64 7}
!18 = !{i64 10}
!19 = !{i64 3}

; IR: define {{.*}} @_Z1Ab(i1 noundef zeroext %b)
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD:[0-9]+]]
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD:[0-9]+]]
; IR: define {{.*}} @_Z2XZv()
; IR:   call {{.*}} @_Z1Ab.memprof.1(i1 noundef zeroext false)
; IR: define {{.*}} @_Z1Mv()
; IR:   call {{.*}} @_Z2XZv()
;; Make sure the inlined context in _Z3XZNv, which partially overlaps
;; trimmed cold context, and also partially overlaps completely
;; unrelated contexts, correctly calls the cloned version of Z1Ab
;; that will call the cold annotated allocation.
; IR: define {{.*}} @_Z3XZNv()
; IR:   call {{.*}} @_Z1Ab.memprof.1(i1 noundef zeroext true)
; IR: define {{.*}} @_Z1Yv()
; IR:   call {{.*}} @_Z1Ab(i1 noundef zeroext true)
; IR: define {{.*}} @main()
; IR:   call {{.*}} @_Z1Mv()
; IR:   call {{.*}} @_Z1Mv.memprof.1()
; IR:   call {{.*}} @_Z3XZNv()
; IR:   call {{.*}} @_Z1Yv()
; IR: define {{.*}} @_Z1Ab.memprof.1(i1 noundef zeroext %b)
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD]]
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD]]
; IR: define {{.*}} @_Z2XZv.memprof.1()
; IR:   call {{.*}} @_Z1Ab(i1 noundef zeroext false)
; IR: define {{.*}} @_Z1Mv.memprof.1()
; IR:   call {{.*}} @_Z2XZv.memprof.1()

; STATS: 2 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS: 2 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS: 3 memprof-context-disambiguation - Number of function clones created during whole program analysis
