; RUN: llc %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-arm-none-eabi"

; CHECK-LABEL: x:
; CHECK: bl OUTLINED_FUNCTION
; CHECK-LABEL: y:
; CHECK: bl OUTLINED_FUNCTION
; CHECK-LABEL: z:
; CHECK-NOT: bl OUTLINED_FUNCTION

@a = hidden global i32 0, align 4
@b = hidden global i32 0, align 4
@c = hidden global i32 0, align 4
@d = hidden global i32 0, align 4
@e = hidden global i32 0, align 4
@f = hidden global i32 0, align 4
@g = hidden global i32 0, align 4
@h = hidden global i32 0, align 4
@i = hidden global i32 0, align 4

define hidden i32 @x() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, ptr @a, align 4
  %1 = load volatile i32, ptr @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, ptr @c, align 4
  %3 = load volatile i32, ptr @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, ptr @e, align 4
  %5 = load volatile i32, ptr @f, align 4
  %6 = load volatile i32, ptr @g, align 4
  %div3 = sdiv i32 %5, %6
  %7 = load volatile i32, ptr @h, align 4
  %8 = load volatile i32, ptr @i, align 4
  %add2 = add i32 %div, 1
  %add4 = add i32 %add2, %4
  %add5 = add i32 %add4, %div3
  %add6 = add i32 %add5, %7
  %add7 = add i32 %add6, %8
  ret i32 %add7
}

define hidden i32 @y() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, ptr @a, align 4
  %1 = load volatile i32, ptr @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, ptr @c, align 4
  %3 = load volatile i32, ptr @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, ptr @e, align 4
  %5 = load volatile i32, ptr @f, align 4
  %6 = load volatile i32, ptr @g, align 4
  %div3 = sdiv i32 %5, %6
  %7 = load volatile i32, ptr @h, align 4
  %8 = load volatile i32, ptr @i, align 4
  %add2 = add i32 %div, 2
  %add4 = add i32 %add2, %4
  %add5 = add i32 %add4, %div3
  %add6 = add i32 %add5, %7
  %add7 = add i32 %add6, %8
  ret i32 %add7
}

define hidden i32 @z() local_unnamed_addr #1 {
entry:
  %0 = load volatile i32, ptr @a, align 4
  %1 = load volatile i32, ptr @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, ptr @c, align 4
  %3 = load volatile i32, ptr @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, ptr @e, align 4
  %5 = load volatile i32, ptr @f, align 4
  %6 = load volatile i32, ptr @g, align 4
  %div3 = sdiv i32 %5, %6
  %7 = load volatile i32, ptr @h, align 4
  %8 = load volatile i32, ptr @i, align 4
  %add2 = add i32 %div, 3
  %add4 = add i32 %add2, %4
  %add5 = add i32 %add4, %div3
  %add6 = add i32 %add5, %7
  %add7 = add i32 %add6, %8
  ret i32 %add7
}

attributes #0 = { minsize nofree norecurse nounwind optsize "sign-return-address"="non-leaf" }
attributes #1 = { minsize nofree norecurse nounwind optsize }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}
