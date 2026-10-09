;; Test that a trimmed context terminating at a node that is cloned
;; again while creating a merge node (during mergeClones) is duplicated onto
;; the merge node, and that the duplicated context ids are known to the
;; allocation node mapping used when deciding whether other callers can share
;; the merge node.
;;
;; The code is that of mergenodes.ll (main calls foo twice, foo->baz->bar,
;; bar has two allocations with opposite cold/notcold behavior from the two
;; main callsites), plus a third allocation in bar that is cold via a trimmed
;; context ending at foo's call to baz (i.e. cold from any caller of foo), and
;; notcold via a separate path main->qux->bar.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-merge-iteration=false \
; RUN:	-memprof-verify-ccg -memprof-verify-nodes -stats \
; RUN:  -pass-remarks=memprof-context-disambiguation %s -S 2>&1 | \
; RUN:  FileCheck %s --check-prefix=IR --check-prefix=STATS --check-prefix=REMARKS

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  ;; Ultimately calls bar and allocates notcold memory from first call to new
  ;; and cold memory from second call to new.
  %call = call noundef ptr @_Z3foov(), !callsite !0
  ;; Ultimately calls bar and allocates cold memory from first call to new
  ;; and notcold memory from second call to new.
  %call1 = call noundef ptr @_Z3foov(), !callsite !1
  ;; Calls bar directly via qux; third allocation is notcold.
  %call2 = call noundef ptr @_Z3quxv(), !callsite !20
  ret i32 0
}

define internal ptr @_Z3barv() {
entry:
  ;; notcold when called from first call to foo from main, cold when called from second.
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0, !memprof !2, !callsite !7
  ;; cold when called from first call to foo from main, notcold when called from second.
  %call2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0, !memprof !13, !callsite !18
  ;; cold via trimmed context ending at foo's call to baz, notcold via qux.
  %call3 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0, !memprof !21, !callsite !26
  ret ptr null
}

declare ptr @_Znam(i64)

define internal ptr @_Z3bazv() {
entry:
  %call = call noundef ptr @_Z3barv(), !callsite !8
  ret ptr null
}

; Function Attrs: noinline
define internal ptr @_Z3foov() {
entry:
  %call = call noundef ptr @_Z3bazv(), !callsite !9
  ret ptr null
}

define internal ptr @_Z3quxv() {
entry:
  %call = call noundef ptr @_Z3barv(), !callsite !27
  ret ptr null
}

attributes #0 = { builtin }

!0 = !{i64 8632435727821051414}
!1 = !{i64 -3421689549917153178}
!2 = !{!3, !5}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!8 = !{i64 -5964873800580613432}
!9 = !{i64 2732490490862098848}
!13 = !{!14, !16}
!14 = !{!15, !"cold"}
!15 = !{i64 123, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!16 = !{!17, !"notcold"}
!17 = !{i64 123, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!18 = !{i64 123}
!20 = !{i64 777}
!21 = !{!22, !24}
;; Trimmed cold context: ends at foo's call to baz (stack id 2732490490862098848).
!22 = !{!23, !"cold"}
!23 = !{i64 321, i64 -5964873800580613432, i64 2732490490862098848}
!24 = !{!25, !"notcold"}
!25 = !{i64 321, i64 555, i64 777}
!26 = !{i64 321}
!27 = !{i64 555}

; REMARKS: created clone _Z3barv.memprof.1
; REMARKS: created clone _Z3barv.memprof.2
; REMARKS: created clone _Z3bazv.memprof.1
; REMARKS: created clone _Z3bazv.memprof.2
; REMARKS: created clone _Z3foov.memprof.1
; REMARKS: created clone _Z3foov.memprof.2

;; The third allocation must be cold in every clone of bar reached via foo
;; (from either call in main), and notcold in the clone reached via qux.
; IR: define {{.*}} @main
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3foov.memprof.2()
; IR-NEXT:   call {{.*}} @_Z3foov.memprof.1()
; IR-NEXT:   call {{.*}} @_Z3quxv()
;; Only reached from qux: the first two allocations have no contexts here.
; IR: define internal {{.*}} @_Z3barv()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[DEFAULT:[0-9]+]]
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[DEFAULT]]
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3bazv()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3barv.memprof.1()
; IR: define internal {{.*}} @_Z3foov()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3bazv()
; IR: define internal {{.*}} @_Z3quxv()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3barv()
; IR: define internal {{.*}} @_Z3barv.memprof.1()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD:[0-9]+]]
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD]]
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD]]
; IR: define internal {{.*}} @_Z3barv.memprof.2()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD]]
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD]]
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD]]
; IR: define internal {{.*}} @_Z3bazv.memprof.1()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3barv.memprof.2()
; IR: define internal {{.*}} @_Z3bazv.memprof.2()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3barv.memprof.1()
; IR: define internal {{.*}} @_Z3foov.memprof.1()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3bazv.memprof.2()
; IR: define internal {{.*}} @_Z3foov.memprof.2()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3bazv.memprof.1()
; IR: attributes #[[DEFAULT]] = { builtin }
; IR: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }

; STATS: 4 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS: 3 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS: 6 memprof-context-disambiguation - Number of function clones created during whole program analysis
;; The context ids duplicated while creating merge nodes must be known to the
;; allocation node map used by the merging.
; STATS-NOT: Number of missing alloc nodes for context ids
