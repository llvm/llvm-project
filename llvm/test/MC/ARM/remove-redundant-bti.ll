; RUN: llc -O2 -mtriple=thumbv7m-eabi < %s | FileCheck %s
; RUN: llc -O2 -mtriple=thumbv8m.main-eabi < %s | FileCheck %s

define dso_local i32 @test_for_tbb_tbh_cases(i32 noundef %x) local_unnamed_addr #0 {
entry:
  switch i32 %x, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %return
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  %call = tail call i32 @g(i32 noundef 0) #2
  %add = add nsw i32 %call, 1
  br label %return

sw.bb2:                                           ; preds = %entry
  br label %return

sw.bb3:                                           ; preds = %entry
  br label %return

sw.default:                                       ; preds = %entry
  br label %return

return:                                           ; preds = %entry, %sw.default, %sw.bb3, %sw.bb2, %sw.bb
  %retval.0 = phi i32 [ 5, %sw.default ], [ 4, %sw.bb3 ], [ 3, %sw.bb2 ], [ %add, %sw.bb ], [ 2, %entry ]
  ret i32 %retval.0

; CHECK: tbb	[pc, r1]
; CHECK-LABEL: .LBB0_3:
; CHECK-NOT: bti
; CHECK-LABEL: .LBB0_4:
; CHECK-NOT: bti
; CHECK-LABEL: .LBB0_5:
; CHECK-NOT: bti
; CHECK-LABEL: .LBB0_6:
; CHECK-NOT: bti
; CHECK-LABEL: .LBB0_7:
; CHECK-NOT: bti
}

declare dso_local i32 @g(i32 noundef) local_unnamed_addr #1

define dso_local i32 @test_for_direct_jump_cases(i32 noundef %x) local_unnamed_addr #0 {
entry:
  switch i32 %x, label %sw.default [
    i32 0, label %sw.bb
    i32 2, label %cleanup
    i32 8, label %sw.bb2
    i32 5, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  tail call void asm sideeffect ".space 140000", ""()
  br label %cleanup

sw.bb2:                                           ; preds = %entry
  br label %cleanup

sw.bb3:                                           ; preds = %entry
  br label %cleanup

sw.default:                                       ; preds = %entry
  br label %cleanup

cleanup:                                          ; preds = %entry, %sw.default, %sw.bb3, %sw.bb2, %sw.bb
  %retval.0 = phi i32 [ 5, %sw.default ], [ 4, %sw.bb3 ], [ 3, %sw.bb2 ], [ 1, %sw.bb ], [ %x, %entry ]
  ret i32 %retval.0

; CHECK: mov	pc, r1
; CHECK-LABEL: .LBB1_3:
; CHECK-NOT: bti
; CHECK-LABEL: .LBB1_4:
; CHECK-NOT: bti
; CHECK-LABEL: .LBB1_5:
; CHECK-NOT: bti
; CHECK-LABEL: .LBB1_6:
; CHECK-NOT: bti
; CHECK-LABEL: .LBB1_7:
; CHECK-NOT: bti
}
