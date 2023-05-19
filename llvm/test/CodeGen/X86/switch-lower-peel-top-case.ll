; RUN: llc -mtriple=x86_64-linux-gnu -stop-after=finalize-isel < %s  | FileCheck %s

define i32 @foo(i32 %n) !prof !1 {
entry:
  switch i32 %n, label %bb_default [
    i32 8, label %bb1
    i32 -8826, label %bb2
    i32 18312, label %bb3
    i32 18568, label %bb4
    i32 129, label %bb5
  ], !prof !2

; CHECK: successors: %[[PEELED_CASE_LABEL:.*]](0x5999999a), %[[PEELED_SWITCH_LABEL:.*]](0x26666666)
; CHECK:    %[[VAL:[0-9]+]]:gr32 = COPY $edi
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri %[[VAL]], 18568, implicit-def $eflags
; CHECK:    JCC_1 %[[PEELED_CASE_LABEL]], 4, implicit $eflags
; CHECK:    JMP_1 %[[PEELED_SWITCH_LABEL]]
; CHECK:  [[PEELED_SWITCH_LABEL]].{{[a-zA-Z0-9.]+}}:
; CHECK:    successors: %[[BB1_LABEL:.*]](0x0206d3a0), %[[BB2_LABEL:.*]](0x7df92c60)
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri %[[VAL]], 18311, implicit-def $eflags
; CHECK:    JCC_1 %[[BB2_LABEL]], 15, implicit $eflags
; CHECK:    JMP_1 %[[BB1_LABEL]]
; CHECK:  [[BB1_LABEL]].{{[a-zA-Z0-9.]+}}:
; CHECK:    successors: %[[CASE2_LABEL:.*]](0x35e50d5b), %[[BB3_LABEL:.*]](0x4a1af2a5)
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri %[[VAL]], -8826, implicit-def $eflags
; CHECK:    JCC_1 %[[CASE2_LABEL]], 4, implicit $eflags
; CHECK:    JMP_1 %[[BB3_LABEL]]
; CHECK:  [[BB3_LABEL]]
; CHECK:    successors: %[[CASE5_LABEL:.*]](0x45d173c8), %[[BB4_LABEL:.*]](0x3a2e8c38)
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri %[[VAL]], 129, implicit-def $eflags
; CHECK:    JCC_1 %[[CASE5_LABEL]], 4, implicit $eflags
; CHECK:    JMP_1 %[[BB4_LABEL]]
; CHECK:  [[BB4_LABEL:.*]].{{[a-zA-Z0-9.]+}}:
; CHECK:    successors: %[[CASE1_LABEL:.*]](0x66666666), %[[DEFAULT_BB_LABEL:.*]](0x1999999a)
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri8 %[[VAL]], 8, implicit-def $eflags
; CHECK:    JCC_1 %[[CASE1_LABEL]], 4, implicit $eflags
; CHECK:    JMP_1 %[[DEFAULT_BB_LABEL]]
; CHECK:  [[BB2_LABEL]].{{[a-zA-Z0-9.]+}}:
; CHECK:    successors: %[[CASE3_LABEL:.*]](0x7fe44107), %[[DEFAULT_BB_LABEL]](0x001bbef9)
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri %[[VAL]], 18312, implicit-def $eflags
; CHECK:    JCC_1 %[[CASE3_LABEL]], 4, implicit $eflags
; CHECK:    JMP_1 %[[DEFAULT_BB_LABEL]]

bb1:
  br label %return
bb2:
  br label %return
bb3:
  br label %return
bb4:
  br label %return
bb5:
  br label %return
bb_default:
  br label %return

return:
  %retval = phi i32 [ 0, %bb_default ], [ 5, %bb5 ], [ 4, %bb4 ], [ 3, %bb3 ], [ 2, %bb2 ], [ 1, %bb1 ]
  ret i32 %retval
}

; Test the peeling of the merged cases value 85 and 86.
define i32 @foo1(i32 %n) !prof !1 {
entry:
  switch i32 %n, label %bb_default [
    i32 -40, label %bb1
    i32 86, label %bb2
    i32 85, label %bb2
    i32 1, label %bb3
    i32 5, label %bb4
    i32 7, label %bb5
    i32 49, label %bb6
  ], !prof !3

; CHECK:   successors: %[[PEELED_CASE_LABEL:.*]](0x59999999), %[[PEELED_SWITCH_LABEL:.*]](0x26666667)
; CHECK:   %[[VAL:[0-9]+]]:gr32 = COPY $edi
; CHECK:   %{{[0-9]+}}:gr32 = ADD32ri8 %{{[0-9]+}}, -85, implicit-def dead $eflags
; CHECK:   %{{[0-9]+}}:gr32 = SUB32ri8 %{{[0-9]+}}, 2, implicit-def $eflags
; CHECK:   JCC_1 %[[PEELED_CASE_LABEL]], 2, implicit $eflags
; CHECK:   JMP_1 %[[PEELED_SWITCH_LABEL]]
; CHECK: [[PEELED_SWITCH_LABEL]].{{[a-zA-Z0-9.]+}}:
; CHECK:    successors: %[[BB1_LABEL:.*]](0x0088888a), %[[BB2_LABEL:.*]](0x7f777776)
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri8 %[[VAL]], 4, implicit-def $eflags
; CHECK:    JCC_1 %[[BB2_LABEL]], 15, implicit $eflags
; CHECK:    JMP_1 %[[BB1_LABEL]]
; CHECK:  [[BB1_LABEL]].{{[a-zA-Z0-9.]+}}:
; CHECK:    successors: %[[CASE4_LABEL:.*]](0x7f775a4f), %[[BB3_LABEL:.*]](0x0088a5b1)
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri8 %[[VAL]], 1, implicit-def $eflags
; CHECK:    JCC_1 %[[CASE4_LABEL]], 4, implicit $eflags
; CHECK:    JMP_1 %[[BB3_LABEL]]
; CHECK:  [[BB3_LABEL]].{{[a-zA-Z0-9.]+}}:
; CHECK:    successors: %[[CASE1_LABEL:.*]](0x66666666), %[[DEFAULT_BB_LABEL:.*]](0x1999999a)
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri8 %[[VAL]], -40, implicit-def $eflags
; CHECK:    JCC_1 %[[CASE1_LABEL]], 4, implicit $eflags
; CHECK:    JMP_1 %[[DEFAULT_BB_LABEL]]
; CHECK:  [[BB2_LABEL]].{{[a-zA-Z0-9.]+}}:
; CHECK:    successors: %[[CASE5_LABEL:.*]](0x00000000), %[[BB4_LABEL:.*]](0x80000000)
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri8 %[[VAL]], 5, implicit-def $eflags
; CHECK:    JCC_1 %[[CASE5_LABEL]], 4, implicit $eflags
; CHECK:    JMP_1 %[[BB4_LABEL]]
; CHECK:  [[BB4_LABEL]].{{[a-zA-Z0-9.]+}}:
; CHECK:    successors: %[[CASE6_LABEL:.*]](0x00000000), %[[BB5_LABEL:.*]](0x80000000)
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri8 %[[VAL]], 7, implicit-def $eflags
; CHECK:    JCC_1 %[[CASE6_LABEL]], 4, implicit $eflags
; CHECK:    JMP_1 %[[BB5_LABEL]]
; CHECK:  [[BB5_LABEL]].{{[a-zA-Z0-9.]+}}:
; CHECK:    successors: %[[CASE7_LABEL:.*]](0x00000000), %[[DEFAULT_BB_LABEL]](0x80000000)
; CHECK:    %{{[0-9]+}}:gr32 = SUB32ri8 %[[VAL]], 49, implicit-def $eflags
; CHECK:    JCC_1 %[[CASE7_LABEL]], 4, implicit $eflags
; CHECK:    JMP_1 %[[DEFAULT_BB_LABEL]]


bb1:
  br label %return
bb2:
  br label %return
bb3:
  br label %return
bb4:
  br label %return
bb5:
  br label %return
bb6:
  br label %return
bb_default:
  br label %return

return:
  %retval = phi i32 [ 0, %bb_default ], [ 6, %bb6 ], [ 5, %bb5 ], [ 4, %bb4 ], [ 3, %bb3 ], [ 2, %bb2 ], [ 1, %bb1 ]
  ret i32 %retval
}
!1 = !{!"function_entry_count", i64 100000}
!2 = !{!"branch_weights", i32 50, i32 100, i32 200, i32 29500, i32 70000, i32 150}
!3 = !{!"branch_weights", i32 50, i32 100, i32 500, i32 69500, i32 29850, i32 0, i32 0, i32 0}

