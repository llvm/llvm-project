; Test that memdep gets invalidated when the analyses it depends on are
; invalidated.
;
; Check AA. AA is stateless, but given an explicit invalidate (abandon) the
; AAManager is invalidated and we must invalidate memdep as well given
; the transitive dependency.
; RUN: opt -disable-output -debug-pass-manager -aa-pipeline='basic-aa' %s 2>&1 \
; RUN:     -passes='require<memdep>,invalidate<aa>,gvn' \
; RUN:     | FileCheck %s --check-prefix=CHECK-AA-INVALIDATE
; CHECK-AA-INVALIDATE: Running pass: RequireAnalysisPass
; CHECK-AA-INVALIDATE: Running analysis: MemoryDependenceAnalysis
; CHECK-AA-INVALIDATE: Running pass: InvalidateAnalysisPass
; CHECK-AA-INVALIDATE: Invalidating analysis: AAManager
; CHECK-AA-INVALIDATE: Invalidating analysis: MemoryDependenceAnalysis
; CHECK-AA-INVALIDATE: Running pass: GVNPass
; CHECK-AA-INVALIDATE: Running analysis: MemoryDependenceAnalysis
;
; Check domtree specifically.
; RUN: opt -disable-output -debug-pass-manager %s 2>&1 \
; RUN:     -passes='require<memdep>,invalidate<domtree>,gvn' \
; RUN:     | FileCheck %s --check-prefix=CHECK-DT-INVALIDATE
; CHECK-DT-INVALIDATE: Running pass: RequireAnalysisPass
; CHECK-DT-INVALIDATE: Running analysis: MemoryDependenceAnalysis
; CHECK-DT-INVALIDATE: Running pass: InvalidateAnalysisPass
; CHECK-DT-INVALIDATE: Invalidating analysis: DominatorTreeAnalysis
; CHECK-DT-INVALIDATE: Invalidating analysis: MemoryDependenceAnalysis
; CHECK-DT-INVALIDATE: Running pass: GVNPass
; CHECK-DT-INVALIDATE: Running analysis: MemoryDependenceAnalysis
;

define void @test_use_domtree(ptr nocapture %bufUInt, ptr nocapture %pattern) nounwind {
entry:
  br label %for.body

for.exit:                                         ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %tmp8.7, %for.body ]
  %arrayidx = getelementptr i32, ptr %bufUInt, i32 %i.01
  %arrayidx5 = getelementptr i32, ptr %pattern, i32 %i.01
  %tmp6 = load i32, ptr %arrayidx5, align 4
  store i32 %tmp6, ptr %arrayidx, align 4
  %tmp8.7 = add i32 %i.01, 8
  %cmp.7 = icmp ult i32 %tmp8.7, 1024
  br i1 %cmp.7, label %for.body, label %for.exit
}

%t = type { i32 }
declare void @foo(ptr)

define void @test_use_aa(ptr noalias %stuff ) {
entry:
  %before = load i32, ptr %stuff

  call void @foo(ptr null)

  %after = load i32, ptr %stuff
  %sum = add i32 %before, %after

  store i32 %sum, ptr %stuff
  ret void
}
