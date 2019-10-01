; Test that memdep gets invalidated when the analyses it depends on are
; invalidated.
;
; Check AA. AA is stateless, there's nothing to invalidate.
; RUN: opt -disable-output -debug-pass-manager -aa-pipeline='basic-aa' %s 2>&1 \
; RUN:     -passes='require<memdep>,invalidate<aa>,gvn' \
; RUN:     | FileCheck %s --check-prefix=CHECK-AA-INVALIDATE
; CHECK-AA-INVALIDATE: Running pass: RequireAnalysisPass
; CHECK-AA-INVALIDATE: Running analysis: MemoryDependenceAnalysis
; CHECK-AA-INVALIDATE: Running pass: InvalidateAnalysisPass
; CHECK-NOT-AA-INVALIDATE: Invalidating analysis: MemoryDependenceAnalysis
; CHECK-AA-INVALIDATE: Running pass: GVN
; CHECK-NOT-AA-INVALIDATE: Running analysis: MemoryDependenceAnalysis
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
; CHECK-DT-INVALIDATE: Running pass: GVN
; CHECK-DT-INVALIDATE: Running analysis: MemoryDependenceAnalysis
;

define void @test_use_domtree(i32* nocapture %bufUInt, i32* nocapture %pattern) nounwind {
entry:
  br label %for.body

for.exit:                                         ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %tmp8.7, %for.body ]
  %arrayidx = getelementptr i32, i32* %bufUInt, i32 %i.01
  %arrayidx5 = getelementptr i32, i32* %pattern, i32 %i.01
  %tmp6 = load i32, i32* %arrayidx5, align 4
  store i32 %tmp6, i32* %arrayidx, align 4
  %tmp8.7 = add i32 %i.01, 8
  %cmp.7 = icmp ult i32 %tmp8.7, 1024
  br i1 %cmp.7, label %for.body, label %for.exit
}

%t = type { i32 }
declare void @foo(i8*)

define void @test_use_aa(%t* noalias %stuff ) {
entry:
  %p = getelementptr inbounds %t, %t* %stuff, i32 0, i32 0
  %before = load i32, i32* %p

  call void @foo(i8* null)

  %after = load i32, i32* %p
  %sum = add i32 %before, %after

  store i32 %sum, i32* %p
  ret void
}
