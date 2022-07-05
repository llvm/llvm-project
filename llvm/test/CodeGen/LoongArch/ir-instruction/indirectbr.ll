; RUN: llc --mtriple=loongarch64 < %s | FileCheck %s

define i32 @indirectbr(ptr %target) nounwind {
; CHECK-LABEL: indirectbr:
; CHECK:       # %bb.0:
; CHECK-NEXT:    jirl $zero, $a0, 0
; CHECK-NEXT:  .LBB0_1: # %test_label
; CHECK-NEXT:    move $a0, $zero
; CHECK-NEXT:    jirl $zero, $ra, 0
  indirectbr ptr %target, [label %test_label]
test_label:
  br label %ret
ret:
  ret i32 0
}

define i32 @indirectbr_with_offset(ptr %a) nounwind {
; CHECK-LABEL: indirectbr_with_offset:
; CHECK:       # %bb.0:
; CHECK-NEXT:    jirl $zero, $a0, 1380
; CHECK-NEXT:  .LBB1_1: # %test_label
; CHECK-NEXT:    move $a0, $zero
; CHECK-NEXT:    jirl $zero, $ra, 0
  %target = getelementptr inbounds i8, ptr %a, i32 1380
  indirectbr ptr %target, [label %test_label]
test_label:
  br label %ret
ret:
  ret i32 0
}
