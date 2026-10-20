;; Test that correct clones are generated and reached when we need to
;; re-merge clone nodes before function assignment.
;;
;; The code is similar to that of basic.ll, but with a second allocation.

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
;; Disable merge iteration for now as it causes spurious diffs due to different
;; iteration order (but the same ultimate hinting of the contexts).
; RUN:  -memprof-merge-iteration=false \
; RUN:	-memprof-verify-ccg -memprof-dump-ccg %s -S 2>&1 | FileCheck %s \
; RUN:  --check-prefix=IR --check-prefix=DUMP

;; Make sure the option to disable merging causes the expected regression.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-merge-clones=false %s -S 2>&1 | FileCheck %s --check-prefix=NOMERGE
;; main should incorrectly call the same clone of foo.
; NOMERGE: define {{.*}} @main
; NOMERGE-NEXT: entry:
; NOMERGE-NEXT: call {{.*}} @_Z3foov.memprof.1()
; NOMERGE-NEXT: call {{.*}} @_Z3foov.memprof.1()

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
  ret i32 0
}

define internal ptr @_Z3barv() {
entry:
  ;; notcold when called from first call to foo from main, cold when called from second.
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0, !memprof !2, !callsite !7
  ;; cold when called from first call to foo from main, notcold when called from second.
  %call2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0, !memprof !13, !callsite !18
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

;; After cloning, each callsite in main calls different clones of foo with
;; different allocaton types, and ditto all the way through the leaf
;; allocation callsites. The single allocation-type clones are shared between
;; the two callsites in main. This would lead to incorrect assignment of
;; the leaf allocations to function clones as is, since we have lost the
;; information that each callsite in main ultimately reaches two allocation
;; callsites with *different* allocation types.
; DUMP: CCG after cloning:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAR1ALLOC1:0x[a-f0-9]+]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR1ALLOC1]] to Caller: [[BAZ1:0x[a-f0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	Clones: [[BAR2ALLOC1:0x[a-f0-9]+]]

; DUMP: Node [[BAZ1]]
; DUMP: 	  %call = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR1ALLOC1]] to Caller: [[BAZ1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[BAR1ALLOC2:0x[a-f0-9]+]] to Caller: [[BAZ1]] AllocTypes: NotCold ContextIds: 4
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ1]] to Caller: [[FOO1:0x[a-f0-9]+]] AllocTypes: NotCold ContextIds: 1 4
; DUMP: 	Clones: [[BAZ2:0x[a-f0-9]+]]

; DUMP: Node [[FOO1]]
; DUMP: 	  %call = call noundef ptr @_Z3bazv()	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ1]] to Caller: [[FOO1]] AllocTypes: NotCold ContextIds: 1 4
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO1]] to Caller: [[MAIN1:0x[a-f0-9]+]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[FOO1]] to Caller: [[MAIN2:0x[a-f0-9]+]] AllocTypes: NotCold ContextIds: 4
; DUMP: 	Clones: [[FOO2:0x[a-f0-9]+]]

; DUMP: Node [[MAIN1]]
; DUMP: 	  %call = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO1]] to Caller: [[MAIN1]] AllocTypes: NotCold ContextIds: 1
; DUMP: 		Edge from Callee [[FOO2:0x[a-f0-9]+]] to Caller: [[MAIN1]] AllocTypes: Cold ContextIds: 3
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	  %call1 = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 2 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO1]] to Caller: [[MAIN2]] AllocTypes: NotCold ContextIds: 4
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	CallerEdges:

; DUMP: Node [[BAR1ALLOC2]]
; DUMP: 	  %call2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 4
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR1ALLOC2]] to Caller: [[BAZ1]] AllocTypes: NotCold ContextIds: 4
; DUMP: 	Clones: [[BAR2ALLOC2:0x[a-f0-9]+]]

; DUMP: Node [[FOO2]]
; DUMP: 	  %call = call noundef ptr @_Z3bazv()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ2]] to Caller: [[FOO2]] AllocTypes: Cold ContextIds: 2 3
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: Cold ContextIds: 2
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN1]] AllocTypes: Cold ContextIds: 3
; DUMP: 	Clone of [[FOO1]]

; DUMP: Node [[BAZ2]]
; DUMP: 	  %call = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR2ALLOC1]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 2
; DUMP: 		Edge from Callee [[BAR2ALLOC2]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 3
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ2]] to Caller: [[FOO2]] AllocTypes: Cold ContextIds: 2 3
; DUMP: 	Clone of [[BAZ1]]

; DUMP: Node [[BAR2ALLOC1]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR2ALLOC1]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 2
; DUMP: 	Clone of [[BAR1ALLOC1]]

; DUMP: Node [[BAR2ALLOC2]]
; DUMP: 	  %call2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 3
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR2ALLOC2]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 3
; DUMP: 	Clone of [[BAR1ALLOC2]]

;; After merging, each callsite in main calls a different single clone of foo
;; with both cold and not cold allocation types, but ultimately reaches two
;; single allocation type allocation callsite clones of the correct
;; combination. The graph after assigning function clones is the same, but
;; with function calls updated to the new function clones.
; DUMP: CCG after merging:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAR1ALLOC1]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR1ALLOC1]] to Caller: [[BAZ2]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	Clones: [[BAR2ALLOC1]]

; DUMP: Node [[MAIN1]]
; DUMP: 	  %call = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO3:0x[a-f0-9]+]] to Caller: [[MAIN1]] AllocTypes: NotColdCold ContextIds: 1 3
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	  %call1 = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 2 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: NotColdCold ContextIds: 2 4
; DUMP: 	CallerEdges:

; DUMP: Node [[BAR1ALLOC2]]
; DUMP: 	  %call2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 4
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR1ALLOC2]] to Caller: [[BAZ3:0x[a-f0-9]+]] AllocTypes: NotCold ContextIds: 4
; DUMP: 	Clones: [[BAR2ALLOC2]]

; DUMP: Node [[FOO2]]
; DUMP: 	  %call = call noundef ptr @_Z3bazv()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 2 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ3]] to Caller: [[FOO2]] AllocTypes: NotColdCold ContextIds: 2 4
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: NotColdCold ContextIds: 2 4
; DUMP: 	Clone of [[FOO1]]

; DUMP: Node [[BAZ2]]
; DUMP: 	  %call = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR2ALLOC2]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 3
; DUMP: 		Edge from Callee [[BAR1ALLOC1]] to Caller: [[BAZ2]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ2]] to Caller: [[FOO3]] AllocTypes: NotColdCold ContextIds: 1 3
; DUMP: 	Clone of [[BAZ1]]

; DUMP: Node [[BAR2ALLOC1]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR2ALLOC1]] to Caller: [[BAZ3]] AllocTypes: Cold ContextIds: 2
; DUMP: 	Clone of [[BAR1ALLOC1]]

; DUMP: Node [[BAR2ALLOC2]]
; DUMP: 	  %call2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 3
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR2ALLOC2]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 3
; DUMP: 	Clone of [[BAR1ALLOC2]]

; DUMP: Node [[FOO3]]
; DUMP: 	  %call = call noundef ptr @_Z3bazv()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ2]] to Caller: [[FOO3]] AllocTypes: NotColdCold ContextIds: 1 3
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO3]] to Caller: [[MAIN1]] AllocTypes: NotColdCold ContextIds: 1 3
; DUMP: 	Clone of [[FOO1]]

; DUMP: Node [[BAZ3]]
; DUMP: 	  %call = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 2 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR2ALLOC1]] to Caller: [[BAZ3]] AllocTypes: Cold ContextIds: 2
; DUMP: 		Edge from Callee [[BAR1ALLOC2]] to Caller: [[BAZ3]] AllocTypes: NotCold ContextIds: 4
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ3]] to Caller: [[FOO2]] AllocTypes: NotColdCold ContextIds: 2 4
; DUMP: 	Clone of [[BAZ1]]

; DUMP: CCG after assigning function clones:
; DUMP: Callsite Context Graph:
; DUMP: Node [[BAR1ALLOC1]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0	(clone 0)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 1
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR1ALLOC1]] to Caller: [[BAZ2]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	Clones: [[BAR2ALLOC1]]

; DUMP: Node [[MAIN1]]
; DUMP: 	  %call = call noundef ptr @_Z3foov.memprof.1()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO3]] to Caller: [[MAIN1]] AllocTypes: NotColdCold ContextIds: 1 3
; DUMP: 	CallerEdges:

; DUMP: Node [[MAIN2]]
; DUMP: 	  %call1 = call noundef ptr @_Z3foov()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 2 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: NotColdCold ContextIds: 2 4
; DUMP: 	CallerEdges:

; DUMP: Node [[BAR1ALLOC2]]
; DUMP: 	  %call2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #1	(clone 1)
; DUMP: 	AllocTypes: NotCold
; DUMP: 	ContextIds: 4
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR1ALLOC2]] to Caller: [[BAZ3]] AllocTypes: NotCold ContextIds: 4
; DUMP: 	Clones: [[BAR2ALLOC2]]

; DUMP: Node [[FOO2]]
; DUMP: 	  %call = call noundef ptr @_Z3bazv.memprof.1()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 2 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ3]] to Caller: [[FOO2]] AllocTypes: NotColdCold ContextIds: 2 4
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO2]] to Caller: [[MAIN2]] AllocTypes: NotColdCold ContextIds: 2 4
; DUMP: 	Clone of [[FOO1]]

; DUMP: Node [[BAZ2]]
; DUMP: 	  %call = call noundef ptr @_Z3barv()	(clone 0)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR2ALLOC2]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 3
; DUMP: 		Edge from Callee [[BAR1ALLOC1]] to Caller: [[BAZ2]] AllocTypes: NotCold ContextIds: 1
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ2]] to Caller: [[FOO3]] AllocTypes: NotColdCold ContextIds: 1 3
; DUMP: 	Clone of [[BAZ1]]

; DUMP: Node [[BAR2ALLOC1]]
; DUMP: 	  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0	(clone 1)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 2
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR2ALLOC1]] to Caller: [[BAZ3]] AllocTypes: Cold ContextIds: 2
; DUMP: 	Clone of [[BAR1ALLOC1]]

; DUMP: Node [[BAR2ALLOC2]]
; DUMP: 	  %call2 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #1	(clone 0)
; DUMP: 	AllocTypes: Cold
; DUMP: 	ContextIds: 3
; DUMP: 	CalleeEdges:
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAR2ALLOC2]] to Caller: [[BAZ2]] AllocTypes: Cold ContextIds: 3
; DUMP: 	Clone of [[BAR1ALLOC2]]

; DUMP: Node [[FOO3]]
; DUMP: 	  %call = call noundef ptr @_Z3bazv()	(clone 1)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 1 3
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAZ2]] to Caller: [[FOO3]] AllocTypes: NotColdCold ContextIds: 1 3
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[FOO3]] to Caller: [[MAIN1]] AllocTypes: NotColdCold ContextIds: 1 3
; DUMP: 	Clone of [[FOO1]]

; DUMP: Node [[BAZ3]]
; DUMP: 	  %call = call noundef ptr @_Z3barv.memprof.1()	(clone 1)
; DUMP: 	AllocTypes: NotColdCold
; DUMP: 	ContextIds: 2 4
; DUMP: 	CalleeEdges:
; DUMP: 		Edge from Callee [[BAR2ALLOC1]] to Caller: [[BAZ3]] AllocTypes: Cold ContextIds: 2
; DUMP: 		Edge from Callee [[BAR1ALLOC2]] to Caller: [[BAZ3]] AllocTypes: NotCold ContextIds: 4
; DUMP: 	CallerEdges:
; DUMP: 		Edge from Callee [[BAZ3]] to Caller: [[FOO2]] AllocTypes: NotColdCold ContextIds: 2 4
; DUMP: 	Clone of [[BAZ1]]

; IR: define {{.*}} @main
;; The first call to foo should now call foo.memprof.1 that ultimately
;; calls bar with the first allocation hinted notcold and the second cold
;; (via call chain foo.memprof.1 -> baz -> bar).
; IR:   call {{.*}} @_Z3foov.memprof.1()
;; The second call to foo still calls the original foo, but ultimately
;; reaches a clone of bar with the first allocation hinted cold and the
;; second notcold.
; IR:   call {{.*}} @_Z3foov()
; IR: define internal {{.*}} @_Z3barv()
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD:[0-9]+]]
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3bazv()
; IR:   call {{.*}} @_Z3barv()
; IR: define internal {{.*}} @_Z3foov()
; IR:   call {{.*}} @_Z3bazv.memprof.1()
; IR: define internal {{.*}} @_Z3barv.memprof.1()
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[COLD]]
; IR:   call {{.*}} @_Znam(i64 noundef 10) #[[NOTCOLD]]
; IR: define internal {{.*}} @_Z3bazv.memprof.1()
; IR:   call {{.*}} @_Z3barv.memprof.1()
; IR: define internal {{.*}} @_Z3foov.memprof.1()
; IR:   call {{.*}} @_Z3bazv()
; IR: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }
