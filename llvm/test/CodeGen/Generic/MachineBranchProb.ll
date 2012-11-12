; RUN: llc < %s -print-machineinstrs=expand-isel-pseudos -o /dev/null 2>&1 | FileCheck %s

; Make sure we have the correct weight attached to each successor.
define i32 @test2(i32 %x) nounwind uwtable readnone ssp {
; CHECK: Machine code for function test2:
entry:
  %conv = sext i32 %x to i64
  switch i64 %conv, label %return [
    i64 0, label %sw.bb
    i64 1, label %sw.bb
    i64 4, label %sw.bb
    i64 5, label %sw.bb1
  ], !prof !0
; CHECK: BB#0: derived from LLVM BB %entry
; CHECK: Successors according to CFG: BB#2(64) BB#4(14)
; CHECK: BB#4: derived from LLVM BB %entry
; CHECK: Successors according to CFG: BB#1(10) BB#5(4)
; CHECK: BB#5: derived from LLVM BB %entry
; CHECK: Successors according to CFG: BB#1(4) BB#3(7)

sw.bb:
  br label %return

sw.bb1:
  br label %return

return:
  %retval.0 = phi i32 [ 5, %sw.bb1 ], [ 1, %sw.bb ], [ 0, %entry ]
  ret i32 %retval.0
}

!0 = metadata !{metadata !"branch_weights", i32 7, i32 6, i32 4, i32 4, i32 64}
