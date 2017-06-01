; RUN: llc -mtriple=arm-apple-ios -print-machineinstrs=branch-folder \
; RUN: %s -o /dev/null 2>&1 | FileCheck %s

; Branch probability of tailed-merged block:
;
; p(L0_L1 -> L2) = p(entry -> L0) * p(L0 -> L2) + p(entry -> L1) * p(L1 -> L2)
;                = 0.2 * 0.6 + 0.8 * 0.3 = 0.36
; p(L0_L1 -> L3) = p(entry -> L0) * p(L0 -> L3) + p(entry -> L1) * p(L1 -> L3)
;                = 0.2 * 0.4 + 0.8 * 0.7 = 0.64

; CHECK: # Machine code for function test0:
; CHECK: Successors according to CFG: BB#{{[0-9]+}}({{[0-9a-fx/= ]+}}20.00%) BB#{{[0-9]+}}({{[0-9a-fx/= ]+}}80.00%)
; CHECK: BB#{{[0-9]+}}:
; CHECK: BB#{{[0-9]+}}:
; CHECK: # End machine code for function test0.

define i32 @test0(i32 %n, i32 %m, i32* nocapture %a, i32* nocapture %b) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %L0, label %L1, !prof !0

L0:                                          ; preds = %entry
  store i32 12, i32* %a, align 4
  store i32 18, i32* %b, align 4
  %cmp1 = icmp eq i32 %m, 8
  br i1 %cmp1, label %L2, label %L3, !prof !1

L1:                                          ; preds = %entry
  store i32 14, i32* %a, align 4
  store i32 18, i32* %b, align 4
  %cmp3 = icmp eq i32 %m, 8
  br i1 %cmp3, label %L2, label %L3, !prof !2

L2:                                               ; preds = %L1, %L0
  br label %L3

L3:                                           ; preds = %L0, %L1, %L2
  %retval.0 = phi i32 [ 100, %L2 ], [ 6, %L1 ], [ 6, %L0 ]
  ret i32 %retval.0
}

!0 = !{!"branch_weights", i32 200, i32 800}
!1 = !{!"branch_weights", i32 600, i32 400}
!2 = !{!"branch_weights", i32 300, i32 700}
