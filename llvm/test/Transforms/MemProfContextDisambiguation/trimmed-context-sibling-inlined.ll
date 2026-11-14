;; Test that a cold callsite whose contexts are trimmed at an interior frame is
;; still cloned correctly when (a) the same caller has a sibling callsite into
;; the same callee that is cold on only one of its paths, so the caller itself
;; must be cloned, and (b) the frame the trimmed context ends at is a real
;; function on one path and inlined into the caller's caller on the other path.
;;
;; Original code looks like:
;;
;;   char *wrapper()   { return new char[10]; }
;;   void apply()      { wrapper() /*c66*/; ...; wrapper() /*c655*/; }
;;   void add()        { apply(); }
;;   void pathA()      { add(); }
;;   void lambdaB()    { apply(); }   // add inlined into lambdaB
;;
;; Profile (stack ids in comments):
;;   wrapper <- apply:c66  <- add <- pathA             notcold  (ids 1 2 5 6)
;;   wrapper <- apply:c66  <- add(inlined) <- lambdaB  cold     (ids 1 2 5 7)
;;   wrapper <- apply:c655 <- add                      cold, trimmed at add (ids 1 3 5)
;;
;; The trimmed c655 context sits on its own callsite node (apply:c655->wrapper),
;; so it is cold in every clone of apply regardless of how add is inlined
;; above it; the sibling c66 callsite is handled by cloning apply for lambdaB.
;;
;; The second RUN drops the !callsite on apply:c655 (as a profile that lost the
;; out-of-line callsite record for that call would). The call can then not be
;; redirected: it keeps calling the original wrapper from both copies of apply
;; and the original allocation retains a cold and a notcold context.

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=IR --check-prefix=REMARKS

; RUN: sed 's/, !callsite !9//' %s | opt -passes=memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation -S 2>&1 \
; RUN:  | FileCheck %s --check-prefix=LOST

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @wrapper() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !0, !callsite !7
  ret ptr %call
}

define void @apply() {
entry:
  %c66 = call ptr @wrapper(), !callsite !8
  %c655 = call ptr @wrapper(), !callsite !9
  ret void
}

define void @add() {
entry:
  call void @apply(), !callsite !10
  ret void
}

define void @pathA() {
entry:
  call void @add(), !callsite !11
  ret void
}

;; add was inlined into lambdaB: the call to apply carries the inlined
;; add->apply frame (5) plus lambdaB's own frame (7).
define void @lambdaB() {
entry:
  call void @apply(), !callsite !12
  ret void
}

declare ptr @_Znam(i64)

attributes #0 = { builtin }

;; Stack ids: 1 = wrapper alloc, 2 = apply:c66->wrapper, 3 = apply:c655->wrapper,
;; 5 = add->apply, 6 = pathA->add, 7 = lambdaB->apply (add inlined).
!0 = !{!1, !3, !5}
!1 = !{!2, !"notcold"}
!2 = !{i64 1, i64 2, i64 5, i64 6}
!3 = !{!4, !"cold"}
!4 = !{i64 1, i64 2, i64 5, i64 7}
;; Cold context of apply:c655, trimmed at add (no pathA / lambdaB frame).
!5 = !{!6, !"cold"}
!6 = !{i64 1, i64 3, i64 5}
!7 = !{i64 1}
!8 = !{i64 2}
!9 = !{i64 3}
!10 = !{i64 5}
!11 = !{i64 6}
!12 = !{i64 5, i64 7}

;; apply:c655 calls the cold wrapper clone in both copies of apply; apply:c66
;; only in the copy reached from lambdaB.
; REMARKS: call in clone wrapper.memprof.1 marked with memprof allocation attribute cold
; REMARKS: call in clone wrapper marked with memprof allocation attribute notcold
; IR: define {{.*}} @apply()
; IR:   call ptr @wrapper()
; IR:   call ptr @wrapper.memprof.1()
; IR: define {{.*}} @lambdaB()
; IR:   call void @apply.memprof.1()
; IR: define {{.*}} @wrapper.memprof.1()
; IR:   call ptr @_Znam(i64 10) #[[COLD:[0-9]+]]
; IR: define {{.*}} @apply.memprof.1()
; IR:   call ptr @wrapper.memprof.1()
; IR:   call ptr @wrapper.memprof.1()
; IR: attributes #{{[0-9]+}} = { builtin "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }

;; Without the apply:c655 callsite metadata that call cannot be redirected.
; LOST: call in clone wrapper.memprof.1 marked with memprof allocation attribute cold
; LOST: call in clone wrapper marked with memprof allocation attribute ambiguous
; LOST: define {{.*}} @apply()
; LOST:   call ptr @wrapper()
; LOST:   call ptr @wrapper()
; LOST: define {{.*}} @wrapper.memprof.1()
; LOST:   call ptr @_Znam(i64 10) #[[COLD:[0-9]+]]
; LOST: define {{.*}} @apply.memprof.1()
; LOST:   call ptr @wrapper.memprof.1()
; LOST:   call ptr @wrapper()
; LOST: attributes #{{[0-9]+}} = { builtin "memprof"="ambiguous" }
; LOST: attributes #[[COLD]] = { builtin "memprof"="cold" }
