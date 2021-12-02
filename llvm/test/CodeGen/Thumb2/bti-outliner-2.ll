; RUN: llc -mtriple=thumbv7m-eabi %s -o - | FileCheck %s

; See bti-outliner-1.ll
; Difference is the BTI placement is enabled by default for the entire module.

@a = hidden global i32 0, align 4
@b = hidden global i32 0, align 4
@c = hidden global i32 0, align 4
@d = hidden global i32 0, align 4
@e = hidden global i32 0, align 4
@f = hidden global i32 0, align 4

define hidden i32 @x(i32 %p) local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, i32* @a, align 4
  %1 = load volatile i32, i32* @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, i32* @c, align 4
  %3 = load volatile i32, i32* @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, i32* @e, align 4
  %mul = mul nsw i32 %4, %div
  %5 = load volatile i32, i32* @f, align 4
  %add2 = add nsw i32 %mul, %5
  %add3 = add nsw i32 %add2, 1
  ret i32 %add3
}
; CHECK-LABEL: x:
; CHECK:       bti
; CHECK:       bl OUTLINED_FUNCTION_0

define hidden i32 @y(i32 %p) local_unnamed_addr #1 {
entry:
  %0 = load volatile i32, i32* @a, align 4
  %1 = load volatile i32, i32* @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, i32* @c, align 4
  %3 = load volatile i32, i32* @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, i32* @e, align 4
  %mul = mul nsw i32 %4, %div
  %5 = load volatile i32, i32* @f, align 4
  %add2 = add nsw i32 %mul, %5
  %add3 = add nsw i32 %add2, 2
  ret i32 %add3
}
; CHECK-LABEL: y:
; CHECK-NOT:   bti
; CHECK-NOT:   bl OUTLINED_FUNCTION

define hidden i32 @z(i32 %p) local_unnamed_addr #2 {
entry:
  %0 = load volatile i32, i32* @a, align 4
  %1 = load volatile i32, i32* @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, i32* @c, align 4
  %3 = load volatile i32, i32* @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, i32* @e, align 4
  %mul = mul nsw i32 %4, %div
  %5 = load volatile i32, i32* @f, align 4
  %add2 = add nsw i32 %mul, %5
  %add3 = add nsw i32 %add2, 3
  ret i32 %add3
}
; CHECK-LABEL: z:
; CHECK:       bti
; CHECK:       bl OUTLINED_FUNCTION_0

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK:       bti

attributes #0 = { minsize nofree norecurse nounwind optsize  }
attributes #1 = { minsize nofree norecurse nounwind optsize "branch-target-enforcement"="false" }
attributes #2 = { minsize nofree norecurse nounwind optsize "branch-target-enforcement"="true"  }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"branch-target-enforcement", i32 1}
