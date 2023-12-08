; RUN: opt -passes=partial-inliner -S < %s | FileCheck %s

; Checks that valid costs are computed for intrinsic calls.
; https://bugs.llvm.org/show_bug.cgi?id=45932


@emit_notes = external global i8, align 2

; CHECK: var_reg_delete
; CHECK-NEXT: bb
; CHECK-NEXT: tail call void @delete_variable_part()
; CHECK-NEXT: ret void

define void @var_reg_delete() {
bb:
  tail call void @delete_variable_part()
  ret void
}

; CHECK: delete_variable_part
; CHECK-NEXT: bb
; CHECK-NEXT: %tmp1.i = tail call i32 @find_variable_location_part()
; CHECK-NEXT: %tmp3.i = icmp sgt i32 %tmp1.i, -1
; CHECK-NEXT: br i1 %tmp3.i, label %bb4.i, label %delete_slot_part.exit

; CHECK: bb4.i
; CHECK-NEXT: %tmp.i.i = load i8, ptr @emit_notes
; CHECK-NEXT:   %tmp1.i.i = icmp ne i8 %tmp.i.i, 0
; CHECK-NEXT:  tail call void @llvm.assume(i1 %tmp1.i.i)
; CHECK-NEXT:  unreachable

; CHECK: delete_slot_part.exit
; CHECK-NEXT: ret void

define void @delete_variable_part() {
bb:
  %tmp1.i = tail call i32 @find_variable_location_part()
  %tmp3.i = icmp sgt i32 %tmp1.i, -1
  br i1 %tmp3.i, label %bb4.i, label %delete_slot_part.exit

bb4.i:
  %tmp.i.i = load i8, ptr @emit_notes, align 2
  %tmp1.i.i = icmp ne i8 %tmp.i.i, 0
  tail call void @llvm.assume(i1 %tmp1.i.i)
  unreachable

delete_slot_part.exit:
  ret void
}

; CHECK: declare i32 @find_variable_location_part
declare i32 @find_variable_location_part()

; CHECK: declare void @llvm.assume(i1 noundef)
declare void @llvm.assume(i1 noundef)
