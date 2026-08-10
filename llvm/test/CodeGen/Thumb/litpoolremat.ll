; RUN: llc < %s -mtriple=thumb-apple-darwin | FileCheck %s

declare void @consume_value(i32) #1

declare i32 @get_value(...) #1

declare void @consume_three_values(i32, i32, i32) #1

; Function Attrs: nounwind uwtable
define void @should_not_spill() #0 {
  tail call void @consume_value(i32 1764) #2
  %1 = tail call i32 (...) @get_value() #2
  %2 = tail call i32 (...) @get_value() #2
  %3 = tail call i32 (...) @get_value() #2
  tail call void @consume_value(i32 %1) #2
  tail call void @consume_value(i32 %2) #2
  tail call void @consume_value(i32 %3) #2
  tail call void @consume_value(i32 1764) #2
  tail call void @consume_three_values(i32 %1, i32 %2, i32 %3) #2
  ret void
}

; CHECK: ldr r0, LCPI0_0
; CHECK-NOT: str r0
; CHECK: bl
; CHECK: ldr r0, LCPI0_0
; CHECK-LABEL: LCPI0_0:
; CHECK-NEXT: .long 1764
