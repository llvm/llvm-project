; RUN: opt -disable-output -passes=print-predicateinfo < %s 2>&1 | FileCheck %s

%1 = type opaque
%0 = type opaque

; Check we can use ssa.copy with unnamed types.

; CHECK-LABEL: bb:
; CHECK: Has predicate info
; CHECK: branch predicate info { TrueEdge: 1 Comparison:  %cmp1 = icmp ne ptr %arg, null Edge: [label %bb,label %bb1], RenamedOp: %arg }
; CHECK-NEXT:  %arg.0 = call ptr @llvm.ssa.copy.p0(ptr %arg)

; CHECK-LABEL: bb1:
; CHECK: Has predicate info
; CHECK-NEXT: branch predicate info { TrueEdge: 0 Comparison:  %cmp2 = icmp ne ptr null, %tmp Edge: [label %bb1,label %bb3], RenamedOp: %tmp }
; CHECK-NEXT: %tmp.0 = call ptr @llvm.ssa.copy.p0(ptr %tmp)

define void @f0(ptr %arg, ptr %tmp) {
bb:
  %cmp1 = icmp ne ptr %arg, null
  br i1 %cmp1, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  %cmp2 = icmp ne ptr null, %tmp
  br i1 %cmp2, label %bb2, label %bb3

bb2:                                              ; preds = %bb
  ret void

bb3:                                              ; preds = %bb
  %u1 = call ptr @fun(ptr %tmp)
  %tmp2 = call ptr @fun(ptr %arg)
  ret void
}

declare ptr @fun(ptr)
