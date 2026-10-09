;; Test that contexts which were trimmed to end at an interior
;; callsite node are preserved when that node is cloned for a different
;; allocation.
;;
;; Original code looks like:
;;
;; char *leafA() { return new char[10]; }
;; char *leafC() { return new char[10]; }
;; void helper() { leafA(); leafC(); }
;; void mid() { helper(); }
;; void other() { helper(); }
;; void top1() { mid(); }
;; void top2() { mid(); }
;;
;; The allocation in leafA is cold whenever it is reached via mid (from both
;; top1 and top2) and not cold via other. Because the behavior is unambiguous
;; from mid's frame on, the cold context is trimmed to end at mid's
;; call to helper: [leafA, helper, mid]. That context is therefore not attached
;; to any caller of mid in the callsite context graph.
;;
;; The allocation in leafC is cold via top1 and not cold via top2, requiring
;; mid (and helper) to be cloned for top1. Without duplicating the trimmed
;; leafA context onto the new mid clone, the clone's path to leafA would carry
;; no context information and would be assigned the callee version from the
;; original helper (which is not cold for leafA via other), losing the cold hint
;; for the leafA allocation when called via top1.
;;
;; The expected result is identical to what we get when the leafA contexts are
;; not trimmed, i.e. both top1 and top2 reach a cold clone of leafA.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -memprof-dump-ccg \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=DUMP --check-prefix=IR \
; RUN:  --check-prefix=STATS --check-prefix=REMARKS

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @leafA() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !0, !callsite !5
  ret ptr %call
}

define ptr @leafC() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !6, !callsite !11
  ret ptr %call
}

define void @helper() {
entry:
  %call = call ptr @leafA(), !callsite !12
  %call1 = call ptr @leafC(), !callsite !13
  ret void
}

define void @mid() {
entry:
  call void @helper(), !callsite !14
  ret void
}

define void @other() {
entry:
  call void @helper(), !callsite !15
  ret void
}

define void @top1() {
entry:
  call void @mid(), !callsite !16
  ret void
}

define void @top2() {
entry:
  call void @mid(), !callsite !17
  ret void
}

declare ptr @_Znam(i64)

attributes #0 = { builtin }

;; Stack ids: 1 = leafA alloc, 2 = helper->leafA, 5 = leafC alloc,
;; 6 = helper->leafC, 7 = mid->helper, 8 = other->helper, 12 = top1->mid,
;; 13 = top2->mid.
!0 = !{!1, !3}
;; Cold context trimmed at mid's call to helper (no top1/top2 frame).
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 7}
!3 = !{!4, !"notcold"}
!4 = !{i64 1, i64 2, i64 8}
!5 = !{i64 1}
!6 = !{!7, !9}
!7 = !{!8, !"cold"}
!8 = !{i64 5, i64 6, i64 7, i64 12}
!9 = !{!10, !"notcold"}
!10 = !{i64 5, i64 6, i64 7, i64 13}
!11 = !{i64 5}
!12 = !{i64 2}
!13 = !{i64 6}
!14 = !{i64 7}
!15 = !{i64 8}
!16 = !{i64 12}
!17 = !{i64 13}

;; The trimmed cold context (id 1) terminates at the original mid->helper node
;; (NodeId 3): it is on its callee edge to the helper->leafA node (NodeId 2) but
;; not on either of its caller edges from top1 (NodeId 7) or top2 (NodeId 8).
; DUMP: CCG before cloning:
; DUMP: Node {{0x[a-z0-9]+}}
; DUMP-NEXT: 	  %call = call ptr @_Znam(i64 10) #0	(clone 0)
; DUMP-NEXT: 	NodeId: 1
; DUMP: Node [[HELPER_LEAFA:0x[a-z0-9]+]]
; DUMP-NEXT: 	  %call = call ptr @leafA()	(clone 0)
; DUMP-NEXT: 	NodeId: 2
; DUMP: Node [[MID:0x[a-z0-9]+]]
; DUMP-NEXT: 	  call void @helper()	(clone 0)
; DUMP-NEXT: 	NodeId: 3
; DUMP-NEXT: 	AllocTypes: NotColdCold
; DUMP-NEXT: 	ContextIds: 1 3 4
; DUMP-NEXT: 	CalleeEdges:
; DUMP-NEXT: 		Edge from Callee [[HELPER_LEAFA]] to Caller: [[MID]] AllocTypes: Cold ContextIds: 1 (Callee NodeId: 2)
; DUMP-NEXT: 		Edge from Callee {{0x[a-z0-9]+}} to Caller: [[MID]] AllocTypes: NotColdCold ContextIds: 3 4 (Callee NodeId: 6)
; DUMP-NEXT: 	CallerEdges:
; DUMP-NEXT: 		Edge from Callee [[MID]] to Caller: [[TOP1:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 3 (Caller NodeId: 7)
; DUMP-NEXT: 		Edge from Callee [[MID]] to Caller: {{0x[a-z0-9]+}} AllocTypes: NotCold ContextIds: 4 (Caller NodeId: 8)

;; After cloning, the clone of the mid->helper node created for top1 (context
;; id 3) should also contain a duplicate (id 5) of the trimmed cold context, and
;; have an edge to the cold clone of the helper->leafA node, which reaches the
;; cold clone of the leafA allocation.
; DUMP: CCG after cloning:
;; Skip past the original (non-clone) nodes.
; DUMP: {{^}}	NodeId: 8{{$}}
; DUMP: Node [[HELPER_LEAFA_CLONE:0x[a-z0-9]+]]
; DUMP-NEXT: 	  %call = call ptr @leafA()	(clone 0)
; DUMP-NEXT: 	NodeId: 9
; DUMP-NEXT: 	AllocTypes: Cold
; DUMP-NEXT: 	ContextIds: 1 5
; DUMP-NEXT: 	CalleeEdges:
; DUMP-NEXT: 		Edge from Callee [[LEAFA_CLONE:0x[a-z0-9]+]] to Caller: [[HELPER_LEAFA_CLONE]] AllocTypes: Cold ContextIds: 1 5 (Callee NodeId: 10)
; DUMP-NEXT: 	CallerEdges:
; DUMP-NEXT: 		Edge from Callee [[HELPER_LEAFA_CLONE]] to Caller: [[MID]] AllocTypes: Cold ContextIds: 1 (Caller NodeId: 3)
; DUMP-NEXT: 		Edge from Callee [[HELPER_LEAFA_CLONE]] to Caller: [[MID_CLONE:0x[a-z0-9]+]] AllocTypes: Cold ContextIds: 5 (Caller NodeId: 11)
; DUMP-NEXT: 	Clone of [[HELPER_LEAFA]]

; DUMP: Node [[LEAFA_CLONE]]
; DUMP-NEXT: 	  %call = call ptr @_Znam(i64 10) #0	(clone 0)
; DUMP-NEXT: 	NodeId: 10
; DUMP-NEXT: 	AllocTypes: Cold
; DUMP-NEXT: 	ContextIds: 1 5

; DUMP: Node [[MID_CLONE]]
; DUMP-NEXT: 	  call void @helper()	(clone 0)
; DUMP-NEXT: 	NodeId: 11
; DUMP-NEXT: 	AllocTypes: Cold
; DUMP-NEXT: 	ContextIds: 3 5
; DUMP-NEXT: 	CalleeEdges:
; DUMP-NEXT: 		Edge from Callee [[HELPER_LEAFA_CLONE]] to Caller: [[MID_CLONE]] AllocTypes: Cold ContextIds: 5 (Callee NodeId: 9)
; DUMP-NEXT: 		Edge from Callee {{0x[a-z0-9]+}} to Caller: [[MID_CLONE]] AllocTypes: Cold ContextIds: 3 (Callee NodeId: 12)
; DUMP-NEXT: 	CallerEdges:
; DUMP-NEXT: 		Edge from Callee [[MID_CLONE]] to Caller: [[TOP1]] AllocTypes: Cold ContextIds: 3 (Caller NodeId: 7)
; DUMP-NEXT: 	Clone of [[MID]]

; REMARKS: created clone leafA.memprof.1
; REMARKS: created clone leafC.memprof.1
; REMARKS: created clone helper.memprof.1
; REMARKS: created clone helper.memprof.2
; REMARKS: created clone mid.memprof.1

;; Both top1 (via the cloned mid) and top2 (via the original mid) must reach
;; the cold clone of leafA, while other reaches the original (not cold) leafA.
; IR: define {{.*}} @leafA()
; IR:   call {{.*}} @_Znam(i64 10) #[[NOTCOLD:[0-9]+]]
; IR: define {{.*}} @leafC()
; IR:   call {{.*}} @_Znam(i64 10) #[[NOTCOLD]]
; IR: define {{.*}} @helper()
; IR:   call {{.*}} @leafA()
; IR:   call {{.*}} @leafC()
; IR: define {{.*}} @mid()
; IR:   call {{.*}} @helper.memprof.1()
; IR: define {{.*}} @other()
; IR:   call {{.*}} @helper()
; IR: define {{.*}} @top1()
; IR:   call {{.*}} @mid.memprof.1()
; IR: define {{.*}} @top2()
; IR:   call {{.*}} @mid()
; IR: define {{.*}} @leafA.memprof.1()
; IR:   call {{.*}} @_Znam(i64 10) #[[COLD:[0-9]+]]
; IR: define {{.*}} @leafC.memprof.1()
; IR:   call {{.*}} @_Znam(i64 10) #[[COLD]]
; IR: define {{.*}} @helper.memprof.1()
; IR:   call {{.*}} @leafA.memprof.1()
; IR:   call {{.*}} @leafC()
; IR: define {{.*}} @helper.memprof.2()
; IR:   call {{.*}} @leafA.memprof.1()
; IR:   call {{.*}} @leafC.memprof.1()
; IR: define {{.*}} @mid.memprof.1()
; IR:   call {{.*}} @helper.memprof.2()
; IR: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }

; STATS: 2 memprof-context-disambiguation - Number of cold static allocations (possibly cloned)
; STATS: 2 memprof-context-disambiguation - Number of not cold static allocations (possibly cloned)
; STATS: 5 memprof-context-disambiguation - Number of function clones created during whole program analysis
