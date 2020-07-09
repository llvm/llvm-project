; RUN: opt < %s -print-predicateinfo 2>&1 | FileCheck %s

%1 = type opaque
%0 = type opaque

; Check we can use ssa.copy with unnamed types.

; CHECK-LABEL: bb:
; CHECK: Has predicate info
; CHECK: branch predicate info { TrueEdge: 1 Comparison:  %cmp1 = icmp ne %0* %arg, null Edge: [label %bb,label %bb1], RenamedOp: %arg }
; CHECK-NEXT:  %arg.0 = call %0* @llvm.ssa.copy.{{.+}}(%0* %arg)

; CHECK-LABEL: bb1:
; CHECK: Has predicate info
; CHECK-NEXT: branch predicate info { TrueEdge: 0 Comparison:  %cmp2 = icmp ne %1* null, %tmp Edge: [label %bb1,label %bb3], RenamedOp: %tmp }
; CHECK-NEXT: %tmp.0 = call %1* @llvm.ssa.copy.{{.+}}(%1* %tmp)

define void @f0(%0* %arg, %1* %tmp) {
bb:
  %cmp1 = icmp ne %0* %arg, null
  br i1 %cmp1, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  %cmp2 = icmp ne %1* null, %tmp
  br i1 %cmp2, label %bb2, label %bb3

bb2:                                              ; preds = %bb
  ret void

bb3:                                              ; preds = %bb
  %u1 = call i8* @fun(%1* %tmp)
  %tmp2 = bitcast %0* %arg to i8*
  ret void
}

declare i8* @fun(%1*)
