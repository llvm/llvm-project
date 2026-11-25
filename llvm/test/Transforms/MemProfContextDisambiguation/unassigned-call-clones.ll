;; Test that the calls in a function clone which did not get its own clone of a
;; callsite node follow the callee assignment of that callsite's original node,
;; for every function clone. The loop recording such calls (UnassignedCallClones)
;; used to increment the function clone number before checking it, recording the
;; calls of function clone k as if they were in function clone k+1, and so
;; skipping function clone k whenever k+1 is the clone number of a clone of the
;; callsite node.
;;
;; Original code looks like:
;;
;; char *G() { return new char[10]; }
;; char *H() { return new char[10]; }
;; char *J() { return new char[10]; }
;; void F() { G(); H(); J(); }
;; void A() { F(); }
;; void B() { F(); }
;; void C() { F(); }
;; void D() { F(); }
;; void E() { G(); }
;;
;; The allocation in G has profiled contexts through F from A, B and C (all
;; cold) and through E (not cold), so F's call to G is assigned to a cold clone
;; of G in every version of F. The allocation in H is cold only via D, which is
;; the first callsite requiring a clone of F (F.memprof.1, for D). Although D
;; does reach G through F in the code, the profile has no context D->F->G for
;; the allocation in G (it was not sampled on that path), so F.memprof.1 has no
;; clone of the F->G callsite node and its call to G must follow the original
;; node's assignment to the cold clone of G. The allocation in J is cold only
;; via C, which already shares the original F with A and B, so C gets a further
;; clone F.memprof.2, with its own clone of the F->G node (clone number 2). With
;; the off-by-one, F.memprof.1's call to G was recorded as if in function clone
;; 2, skipped because of that F->G node clone, and left calling the original
;; (not cold) G.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=IR \
; RUN:  --check-prefix=STATS --check-prefix=REMARKS

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @G() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !0, !callsite !9
  ret ptr %call
}

define ptr @H() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !10, !callsite !19
  ret ptr %call
}

define ptr @J() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !20, !callsite !27
  ret ptr %call
}

define void @F() {
entry:
  %call = call ptr @G(), !callsite !28
  %call1 = call ptr @H(), !callsite !29
  %call2 = call ptr @J(), !callsite !30
  ret void
}

define void @A() {
entry:
  call void @F(), !callsite !31
  ret void
}

define void @B() {
entry:
  call void @F(), !callsite !32
  ret void
}

define void @C() {
entry:
  call void @F(), !callsite !33
  ret void
}

define void @D() {
entry:
  call void @F(), !callsite !34
  ret void
}

define void @E() {
entry:
  %call = call ptr @G(), !callsite !35
  ret void
}

declare ptr @_Znam(i64)

attributes #0 = { builtin }

;; Stack ids: 1 = G alloc, 2 = F->G, 3 = E->G, 4 = H alloc, 5 = F->H,
;; 6 = J alloc, 7 = F->J, 10 = A->F, 11 = B->F, 12 = C->F, 13 = D->F.
!0 = !{!1, !3, !5, !7}
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 10}
!3 = !{!4, !"cold"}
!4 = !{i64 1, i64 2, i64 11}
!5 = !{!6, !"cold"}
!6 = !{i64 1, i64 2, i64 12}
!7 = !{!8, !"notcold"}
!8 = !{i64 1, i64 3}
!9 = !{i64 1}
!10 = !{!11, !13, !15, !17}
!11 = !{!12, !"notcold"}
!12 = !{i64 4, i64 5, i64 10}
!13 = !{!14, !"notcold"}
!14 = !{i64 4, i64 5, i64 11}
!15 = !{!16, !"notcold"}
!16 = !{i64 4, i64 5, i64 12}
!17 = !{!18, !"cold"}
!18 = !{i64 4, i64 5, i64 13}
!19 = !{i64 4}
!20 = !{!21, !23, !25}
!21 = !{!22, !"notcold"}
!22 = !{i64 6, i64 7, i64 10}
!23 = !{!24, !"notcold"}
!24 = !{i64 6, i64 7, i64 11}
!25 = !{!26, !"cold"}
!26 = !{i64 6, i64 7, i64 12}
!27 = !{i64 6}
!28 = !{i64 2}
!29 = !{i64 5}
!30 = !{i64 7}
!31 = !{i64 10}
!32 = !{i64 11}
!33 = !{i64 12}
!34 = !{i64 13}
!35 = !{i64 3}

; REMARKS: created clone G.memprof.1
; REMARKS: created clone H.memprof.1
; REMARKS: created clone J.memprof.1
; REMARKS: created clone F.memprof.1
; REMARKS: created clone F.memprof.2

;; Every version of F must call the cold clone of G, including F.memprof.1,
;; which has no clone of the F->G callsite node.
; IR: define {{.*}} @G()
; IR:   call {{.*}} @_Znam(i64 10) #[[NOTCOLD:[0-9]+]]
; IR: define {{.*}} @H()
; IR:   call {{.*}} @_Znam(i64 10) #[[NOTCOLD]]
; IR: define {{.*}} @J()
; IR:   call {{.*}} @_Znam(i64 10) #[[NOTCOLD]]
; IR: define {{.*}} @F()
; IR:   call {{.*}} @G.memprof.1()
; IR:   call {{.*}} @H()
; IR:   call {{.*}} @J()
; IR: define {{.*}} @A()
; IR:   call {{.*}} @F()
; IR: define {{.*}} @B()
; IR:   call {{.*}} @F()
; IR: define {{.*}} @C()
; IR:   call {{.*}} @F.memprof.2()
; IR: define {{.*}} @D()
; IR:   call {{.*}} @F.memprof.1()
; IR: define {{.*}} @E()
; IR:   call {{.*}} @G()
; IR: define {{.*}} @G.memprof.1()
; IR:   call {{.*}} @_Znam(i64 10) #[[COLD:[0-9]+]]
; IR: define {{.*}} @H.memprof.1()
; IR:   call {{.*}} @_Znam(i64 10) #[[COLD]]
; IR: define {{.*}} @J.memprof.1()
; IR:   call {{.*}} @_Znam(i64 10) #[[COLD]]
; IR: define {{.*}} @F.memprof.1()
; IR:   call {{.*}} @G.memprof.1()
; IR:   call {{.*}} @H.memprof.1()
; IR:   call {{.*}} @J()
; IR: define {{.*}} @F.memprof.2()
; IR:   call {{.*}} @G.memprof.1()
; IR:   call {{.*}} @H()
; IR:   call {{.*}} @J.memprof.1()
; IR: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }

; STATS: 3 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS: 3 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS: 5 memprof-context-disambiguation - Number of function clones created during whole program analysis
