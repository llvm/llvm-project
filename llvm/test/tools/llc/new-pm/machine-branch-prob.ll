; RUN: llc -mtriple=x86_64-linux-gnu %s -stop-after=finalize-isel -o - | \
; RUN: llc -mtriple=x86_64-linux-gnu -passes='print<machine-branch-prob>' -x mir -filetype=null 2>&1 | \
; RUN: FileCheck %s

declare void @foo()

define i32 @test2(i32 %x) nounwind uwtable readnone ssp {
entry:
  %conv = sext i32 %x to i64
  switch i64 %conv, label %return [
    i64 0, label %sw.bb
    i64 1, label %sw.bb
    i64 4, label %sw.bb
    i64 5, label %sw.bb1
    i64 15, label %sw.bb
  ], !prof !0

sw.bb:
; this call will prevent simplifyCFG from optimizing the block away in ARM/AArch64.
  tail call void @foo()
  br label %return

sw.bb1:
  br label %return

return:
  %retval.0 = phi i32 [ 5, %sw.bb1 ], [ 1, %sw.bb ], [ 0, %entry ]
  ret i32 %retval.0
}

!0 = !{!"branch_weights", i32 7, i32 6, i32 4, i32 4, i32 64, i21 1000}

; CHECK: Printing analysis 'Machine Branch Probability Analysis' for machine function 'test2':
; CHECK: edge %bb.4 -> %bb.6 probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
; CHECK: edge %bb.5 -> %bb.6 probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
