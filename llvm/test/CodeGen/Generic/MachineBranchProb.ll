; RUN: llc < %s -print-machineinstrs=expand-isel-pseudos -o /dev/null 2>&1 | FileCheck %s

; Hexagon runs passes that renumber the basic blocks, causing this test
; to fail.
; XFAIL: hexagon

; Bug: PR31899
; XFAIL: avr

declare void @foo()

; Make sure we have the correct weight attached to each successor.
define i32 @test2(i32 %x) nounwind uwtable readnone ssp {
; CHECK-LABEL: Machine code for function test2:
entry:
  %conv = sext i32 %x to i64
  switch i64 %conv, label %return [
    i64 0, label %sw.bb
    i64 1, label %sw.bb
    i64 4, label %sw.bb
    i64 5, label %sw.bb1
    i64 15, label %sw.bb
  ], !prof !0
; CHECK: %bb.0: derived from LLVM BB %entry
; CHECK: Successors according to CFG: %bb.1({{[0-9a-fx/= ]+}}92.17%) %bb.4({{[0-9a-fx/= ]+}}7.83%)
; CHECK: %bb.4: derived from LLVM BB %entry
; CHECK: Successors according to CFG: %bb.2({{[0-9a-fx/= ]+}}75.29%) %bb.5({{[0-9a-fx/= ]+}}24.71%)
; CHECK: %bb.5: derived from LLVM BB %entry
; CHECK: Successors according to CFG: %bb.1({{[0-9a-fx/= ]+}}47.62%) %bb.6({{[0-9a-fx/= ]+}}52.38%)
; CHECK: %bb.6: derived from LLVM BB %entry
; CHECK: Successors according to CFG: %bb.1({{[0-9a-fx/= ]+}}36.36%) %bb.3({{[0-9a-fx/= ]+}}63.64%)

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


declare void @g(i32)
define void @left_leaning_weight_balanced_tree(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 0,  label %bb0
    i32 100, label %bb1
    i32 200, label %bb2
    i32 300, label %bb3
    i32 400, label %bb4
    i32 500, label %bb5
  ], !prof !1
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 2) br label %return
bb3: tail call void @g(i32 3) br label %return
bb4: tail call void @g(i32 4) br label %return
bb5: tail call void @g(i32 5) br label %return
return: ret void

; Check that we set branch weights on the pivot cmp instruction correctly.
; Cases {0,10,20,30} go on the left with weight 13; cases {40,50} go on the
; right with weight 20.
;
; CHECK-LABEL: Machine code for function left_leaning_weight_balanced_tree:
; CHECK: %bb.0: derived from LLVM BB %entry
; CHECK-NOT: Successors
; CHECK: Successors according to CFG: %bb.8({{[0-9a-fx/= ]+}}39.71%) %bb.9({{[0-9a-fx/= ]+}}60.29%)
}

!1 = !{!"branch_weights",
  ; Default:
  i32 1,
  ; Case 0, 100, 200:
  i32 10, i32 1, i32 1,
  ; Case 300, 400, 500:
  i32 1, i32 10, i32 10}
