; RUN: llc -mtriple=thumbv7m-eabi %s -o - | FileCheck %s

; See `bti-outliner-cost-1.ll`

@a = hidden global i32 0, align 4
@b = hidden global i32 0, align 4
@c = hidden global i32 0, align 4
@d = hidden global i32 0, align 4
@e = hidden global i32 0, align 4

define hidden i32 @x(i32 %p) local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, ptr @a, align 4
  %1 = load volatile i32, ptr @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, ptr @c, align 4
  %3 = load volatile i32, ptr @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, ptr @e, align 4
  %mul = mul nsw i32 %4, %div
  %add2 = add nsw i32 %mul, 1
  ret i32 %add2
}
; CHECK-LABEL: x:
; CHECK-NOT:   bl OUTLINED_FUNCTION

define hidden i32 @y(i32 %p) local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, ptr @a, align 4
  %1 = load volatile i32, ptr @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, ptr @c, align 4
  %3 = load volatile i32, ptr @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, ptr @e, align 4
  %mul = mul nsw i32 %4, %div
  %add2 = add nsw i32 %mul, 2
  ret i32 %add2
}
; CHECK-LABEL: y:
; CHECK-NOT:   bl OUTLINED_FUNCTION

; CHECK-NOT: OUTLINED_FUNCTION

attributes #0 = { minsize nofree norecurse nounwind optsize "branch-target-enforcement" }
