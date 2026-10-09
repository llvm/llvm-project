;; Test that matching calls (calls in the same function with identical callsite
;; metadata, recorded on a single context node) are remapped correctly when
;; their function is cloned, including when the callsite node is cloned again
;; during function assignment because its caller was moved to a new function
;; clone (which rebuilds the clone's calls from the original ones through the
;; function clone's call map).
;;
;; The code is that of mergenodes-trimmed.ll with bar's allocations moved into
;; leaf functions (so that bar's nodes are callsites), and with the third leaf
;; called twice from bar with the same callsite id.

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
  %call = call noundef ptr @_Z3foov(), !callsite !0
  %call1 = call noundef ptr @_Z3foov(), !callsite !1
  %call2 = call noundef ptr @_Z3quxv(), !callsite !20
  ret i32 0
}

define internal ptr @_Z4leafav() {
entry:
  ;; notcold when called from first call to foo from main, cold when called from second.
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0, !memprof !2, !callsite !7
  ret ptr null
}

define internal ptr @_Z4leafbv() {
entry:
  ;; cold when called from first call to foo from main, notcold when called from second.
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0, !memprof !13, !callsite !18
  ret ptr null
}

define internal ptr @_Z4leafcv() {
entry:
  ;; cold via trimmed context ending at foo's call to baz, notcold via qux.
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0, !memprof !21, !callsite !26
  ret ptr null
}

declare ptr @_Znam(i64)

define internal ptr @_Z3barv() {
entry:
  %call = call noundef ptr @_Z4leafav(), !callsite !30
  %call2 = call noundef ptr @_Z4leafbv(), !callsite !31
  ;; Two calls with the same callsite id: matching calls on one node.
  %call3 = call noundef ptr @_Z4leafcv(), !callsite !32
  %call4 = call noundef ptr @_Z4leafcv(), !callsite !32
  ret ptr null
}

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
!4 = !{i64 9086428284934609951, i64 30, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 30, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!8 = !{i64 -5964873800580613432}
!9 = !{i64 2732490490862098848}
!13 = !{!14, !16}
!14 = !{!15, !"cold"}
!15 = !{i64 123, i64 31, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!16 = !{!17, !"notcold"}
!17 = !{i64 123, i64 31, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!18 = !{i64 123}
!20 = !{i64 777}
!21 = !{!22, !24}
;; Trimmed cold context: ends at foo's call to baz (stack id 2732490490862098848).
!22 = !{!23, !"cold"}
!23 = !{i64 321, i64 32, i64 -5964873800580613432, i64 2732490490862098848}
!24 = !{!25, !"notcold"}
!25 = !{i64 321, i64 32, i64 555, i64 777}
!26 = !{i64 321}
!27 = !{i64 555}
!30 = !{i64 30}
!31 = !{i64 31}
!32 = !{i64 32}

; REMARKS: created clone _Z4leafav.memprof.1
; REMARKS: created clone _Z4leafbv.memprof.1
; REMARKS: created clone _Z4leafcv.memprof.1
; REMARKS: created clone _Z3barv.memprof.1
; REMARKS: created clone _Z3barv.memprof.2
; REMARKS: created clone _Z3bazv.memprof.1
; REMARKS: created clone _Z3bazv.memprof.2
; REMARKS: created clone _Z3foov.memprof.1
; REMARKS: created clone _Z3foov.memprof.2

;; Each version of bar must call the same version of leafc from both matching
;; calls: the cold one from the versions reached via foo (the trimmed context),
;; the notcold one from the version reached via qux.
; IR: define {{.*}} @main
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3foov.memprof.2()
; IR-NEXT:   call {{.*}} @_Z3foov.memprof.1()
; IR-NEXT:   call {{.*}} @_Z3quxv()
; IR: define internal {{.*}} @_Z4leafav()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD:[0-9]+]]
; IR: define internal {{.*}} @_Z4leafbv()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD]]
; IR: define internal {{.*}} @_Z4leafcv()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD]]
; IR: define internal {{.*}} @_Z3barv()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z4leafav()
; IR-NEXT:   call {{.*}} @_Z4leafbv()
; IR-NEXT:   call {{.*}} @_Z4leafcv()
; IR-NEXT:   call {{.*}} @_Z4leafcv()
; IR: define internal {{.*}} @_Z3bazv()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3barv.memprof.1()
; IR: define internal {{.*}} @_Z3foov()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3bazv()
; IR: define internal {{.*}} @_Z3quxv()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z3barv()
; IR: define internal {{.*}} @_Z4leafav.memprof.1()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD:[0-9]+]]
; IR: define internal {{.*}} @_Z4leafbv.memprof.1()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD]]
; IR: define internal {{.*}} @_Z4leafcv.memprof.1()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD]]
; IR: define internal {{.*}} @_Z3barv.memprof.1()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z4leafav.memprof.1()
; IR-NEXT:   call {{.*}} @_Z4leafbv()
; IR-NEXT:   call {{.*}} @_Z4leafcv.memprof.1()
; IR-NEXT:   call {{.*}} @_Z4leafcv.memprof.1()
; IR: define internal {{.*}} @_Z3barv.memprof.2()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @_Z4leafav()
; IR-NEXT:   call {{.*}} @_Z4leafbv.memprof.1()
; IR-NEXT:   call {{.*}} @_Z4leafcv.memprof.1()
; IR-NEXT:   call {{.*}} @_Z4leafcv.memprof.1()
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
; IR: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }

; STATS: 3 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS: 3 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS: 9 memprof-context-disambiguation - Number of function clones created during whole program analysis
; STATS-NOT: Number of missing alloc nodes for context ids
