; When an SCC got split due to inlining, we have two mechanisms for reprocessing the updated SCC, first is UR.UpdatedC
; that repeatedly rerun the new, current SCC; second is a worklist for all newly split SCCs. We need to avoid rerun of
; the same SCC when the SCC is set to be processed by both mechanisms back to back. In pathological cases, such extra,
; redundant rerun could cause exponential size growth due to inlining along cycles.
;
; The test cases here illustrates potential redundant rerun and how it's prevented, however there's no extra inlining
; even if we allow the redundant rerun. In real code, when inliner makes different decisions for different call sites
; of the same caller-callee edge, we could end up getting more recursive inlining without SCC mutation.
;
; REQUIRES: asserts
; RUN: opt < %s -passes='cgscc(inline)' -inline-threshold=500 -debug-only=cgscc -S 2>&1 | FileCheck %s

; CHECK: Running an SCC pass across the RefSCC: [(test1_c, test1_a, test1_b)]
; CHECK: Enqueuing the existing SCC in the worklist:(test1_b)
; CHECK: Enqueuing a newly formed SCC:(test1_c)
; CHECK: Enqueuing a new RefSCC in the update worklist: [(test1_b)]
; CHECK: Switch an internal ref edge to a call edge from 'test1_a' to 'test1_c'
; CHECK: Switch an internal ref edge to a call edge from 'test1_a' to 'test1_a'
; CHECK: Re-running SCC passes after a refinement of the current SCC: (test1_c, test1_a)
; CHECK: Skipping redundant run on SCC: (test1_c, test1_a)
; CHECK: Skipping an SCC that is now part of some other RefSCC...

declare void @external(i32 %seed)

define void @test1_a(i32 %num) {
entry:
  call void @test1_b(i32 %num)
  call void @external(i32 %num)
  ret void
}

define void @test1_b(i32 %num) {
entry:
  call void @test1_c(i32 %num)
  call void @test1_a(i32 %num)
  call void @external(i32 %num)
  ret void
}

define void @test1_c(i32 %num) #0 {
  call void @test1_a(i32 %num)
  ret void
}

attributes #0 = { noinline nounwind optnone }