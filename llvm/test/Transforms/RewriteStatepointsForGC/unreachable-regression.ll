; RUN: opt -S -passes=rewrite-statepoints-for-gc < %s | FileCheck %s
;
; Regression test:
;   After the rewritable callsite collection if any callsite was found
;   in a block that was reported unreachable by DominanceTree then
;   removeUnreachableBlocks() was called. But it is stronger than
;   DominatorTree::isReachableFromEntry(), i.e. removeUnreachableBlocks
;   can remove some blocks for which isReachableFromEntry() returns true.
;   This resulted in stale pointers to the collected but removed
;   callsites. Such stale pointers caused crash when accessed.
declare void @f(ptr addrspace(1) %obj)

define void @test(ptr addrspace(1) %arg) gc "statepoint-example" {
; CHECK-LABEL: test(
; CHECK-NEXT: @f
 call void @f(ptr addrspace(1) %arg) #1
 br i1 true, label %not_zero, label %zero

not_zero:
 ret void

; This block is reachable but removed by removeUnreachableBlocks()
zero:
; CHECK-NOT: @f
 call void @f(ptr addrspace(1) %arg) #1
 ret void

unreach:
 call void @f(ptr addrspace(1) %arg) #1
 ret void
}

attributes #1 = { norecurse noimplicitfloat }
