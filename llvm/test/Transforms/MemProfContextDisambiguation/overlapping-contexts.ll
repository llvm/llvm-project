;; This test ensures that the logic which assigns calls to stack nodes
;; correctly handles cloning of a callsite for a trimmed cold context
;; that partially overlaps with a longer context for a different allocation.

;; The profile data and call stacks were all manually added, but the code
;; would be structured something like the following (fairly contrived to
;; result in the type of control flow needed to test):

;; void A(bool b) {
;;   if (b)
;;     // cold: stack ids 10, 12, 13, 15 (trimmed ids 19, 20)
;;     // not cold: stack ids 10, 12, 13, 14 (trimmed id 21)
;;     new char[10]; // stack id 10
;;   else
;;     // not cold: stack ids 11, 12, 13, 15, 16, 17 (trimmed id 22)
;;     // cold: stack ids 11, 12, 13, 15, 16, 18 (trimmed id 23)
;;     new char[10]; // stack id 11
;; }
;;
;; void X(bool b) {
;;   A(b); // stack ids 12
;; }
;;
;; void B(bool b) {
;;   X(b); // stack id 13
;; }
;;
;; void D() {
;;   B(true); // stack id 14
;; }
;;
;; void C(bool b) {
;;   B(b); // stack id 15
;; }
;;
;; void E(bool b) {
;;   C(b); // stack id 16
;; }
;;
;; void F() {
;;   E(false); // stack id 17
;; }
;;
;; void G() {
;;   E(false); // stack id 18
;; }
;;
;; void M() {
;;   C(true); // stack id 19
;; }
;;
;; int main() {
;;   D(); // stack id 20 (leads to not cold allocation)
;;   M(); // stack id 21 (leads to cold allocation)
;;   F(); // stack id 22 (leads to not cold allocation)
;;   G(); // stack id 23 (leads to cold allocation)
;; }

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:	%s -S 2>&1 | FileCheck %s --check-prefix=IR \
; RUN:  --check-prefix=STATS --check-prefix=REMARKS

; REMARKS: created clone _Z1Ab.memprof.1
; REMARKS: created clone _Z1Xb.memprof.1
; REMARKS: created clone _Z1Bb.memprof.1
; REMARKS: created clone _Z1Cb.memprof.1
; REMARKS: created clone _Z1Eb.memprof.1
; REMARKS: call in clone _Z1Gv assigned to call function clone _Z1Eb.memprof.1
; REMARKS: call in clone _Z1Eb.memprof.1 assigned to call function clone _Z1Cb.memprof.1
;; If we don't perform cloning for each allocation separately, we will miss
;; cloning _Z1Cb for the trimmed cold allocation context leading to the
;; allocation at stack id 10.
; REMARKS: call in clone _Z1Cb.memprof.1 assigned to call function clone _Z1Bb.memprof.1
; REMARKS: call in clone _Z1Fv assigned to call function clone _Z1Eb
; REMARKS: call in clone _Z1Eb assigned to call function clone _Z1Cb
; REMARKS: call in clone _Z1Cb assigned to call function clone _Z1Bb.memprof.1
; REMARKS: call in clone _Z1Bb.memprof.1 assigned to call function clone _Z1Xb.memprof.1
; REMARKS: call in clone _Z1Xb.memprof.1 assigned to call function clone _Z1Ab.memprof.1
; REMARKS: call in clone _Z1Ab.memprof.1 marked with memprof allocation attribute cold
; REMARKS: call in clone _Z1Bb.memprof.1 assigned to call function clone _Z1Xb
; REMARKS: call in clone _Z1Dv assigned to call function clone _Z1Bb
; REMARKS: call in clone _Z1Bb assigned to call function clone _Z1Xb
; REMARKS: call in clone _Z1Xb assigned to call function clone _Z1Ab
; REMARKS: call in clone _Z1Ab marked with memprof allocation attribute notcold
; REMARKS: call in clone _Z1Ab.memprof.1 marked with memprof allocation attribute cold
; REMARKS: call in clone _Z1Ab marked with memprof allocation attribute notcold


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z1Ab(i1 noundef zeroext %b) {
entry:
  br i1 %b, label %if.then, label %if.else

if.then:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7, !memprof !0, !callsite !10
  br label %if.end

if.else:
  %call2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #7, !memprof !5, !callsite !11
  br label %if.end

if.end:
  ret void
}

; Function Attrs: nobuiltin
declare ptr @_Znam(i64) #0

define dso_local void @_Z1Xb(i1 noundef zeroext %b) {
entry:
  tail call void @_Z1Ab(i1 noundef zeroext %b), !callsite !12
  ret void
}

define dso_local void @_Z1Bb(i1 noundef zeroext %b) {
entry:
  tail call void @_Z1Xb(i1 noundef zeroext %b), !callsite !13
  ret void
}

define dso_local void @_Z1Dv() {
entry:
  tail call void @_Z1Bb(i1 noundef zeroext true), !callsite !14
  ret void
}

define dso_local void @_Z1Cb(i1 noundef zeroext %b) {
entry:
  tail call void @_Z1Bb(i1 noundef zeroext %b), !callsite !15
  ret void
}

define dso_local void @_Z1Eb(i1 noundef zeroext %b) {
entry:
  tail call void @_Z1Cb(i1 noundef zeroext %b), !callsite !16
  ret void
}

define dso_local void @_Z1Fv() {
entry:
  tail call void @_Z1Eb(i1 noundef zeroext false), !callsite !17
  ret void
}

define dso_local void @_Z1Gv() {
entry:
  tail call void @_Z1Eb(i1 noundef zeroext false), !callsite !18
  ret void
}

define dso_local void @_Z1Mv() {
entry:
  tail call void @_Z1Cb(i1 noundef zeroext true), !callsite !19
  ret void
}

define dso_local noundef i32 @main() local_unnamed_addr {
entry:
  tail call void @_Z1Dv(), !callsite !20 ;; Not cold context
  tail call void @_Z1Mv(), !callsite !21 ;; Cold context
  tail call void @_Z1Fv(), !callsite !22 ;; Not cold context
  tail call void @_Z1Gv(), !callsite !23 ;; Cold context
  ret i32 0
}

attributes #0 = { nobuiltin }
attributes #7 = { builtin }

!0 = !{!1, !3}
;; Cold (trimmed) context via call to _Z1Dv in main
!1 = !{!2, !"cold"}
!2 = !{i64 10, i64 12, i64 13, i64 15}
;; Not cold (trimmed) context via call to _Z1Mv in main
!3 = !{!4, !"notcold"}
!4 = !{i64 10, i64 12, i64 13, i64 14}
!5 = !{!6, !8}
;; Not cold (trimmed) context via call to _Z1Fv in main
!6 = !{!7, !"notcold"}
!7 = !{i64 11, i64 12, i64 13, i64 15, i64 16, i64 17}
;; Cold (trimmed) context via call to _Z1Gv in main
!8 = !{!9, !"cold"}
!9 = !{i64 11, i64 12, i64 13, i64 15, i64 16, i64 18}
!10 = !{i64 10}
!11 = !{i64 11}
!12 = !{i64 12}
!13 = !{i64 13}
!14 = !{i64 14}
!15 = !{i64 15}
!16 = !{i64 16}
!17 = !{i64 17}
!18 = !{i64 18}
!19 = !{i64 19}
!20 = !{i64 20}
!21 = !{i64 21}
!22 = !{i64 22}
!23 = !{i64 23}

; IR: define {{.*}} @_Z1Cb(i1 noundef zeroext %b)
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z1Bb.memprof.1(i1 noundef zeroext %b)

; IR: define {{.*}} @_Z1Ab.memprof.1(i1 noundef zeroext %b)
; IR-NEXT: entry:
; IR-NEXT:   br i1 %b, label %if.then, label %if.else
; IR-EMPTY:
; IR-NEXT: if.then:
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD:[0-9]+]]
; IR-NEXT:   br label %if.end
; IR-EMPTY:
; IR-NEXT: if.else:
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD]]

; IR: define {{.*}} @_Z1Xb.memprof.1(i1 noundef zeroext %b)
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z1Ab.memprof.1(i1 noundef zeroext %b)

; IR: define {{.*}} @_Z1Bb.memprof.1(i1 noundef zeroext %b)
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z1Xb.memprof.1(i1 noundef zeroext %b)

; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }

; STATS: 2 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS: 2 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS: 5 memprof-context-disambiguation - Number of function clones created during whole program analysis
