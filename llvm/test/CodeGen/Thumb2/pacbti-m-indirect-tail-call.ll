; RUN: llc %s -o - | FileCheck %s --check-prefix=CHECK1
; RUN: llc %s -o - | FileCheck %s --check-prefix=CHECK2
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-unknown-eabi"

@p = hidden local_unnamed_addr global ptr null, align 4

define hidden i32 @f(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %call = tail call i32 @g(i32 %a) #0
  %0 = load ptr, ptr @p, align 4
  %call1 = tail call i32 %0(i32 %call, i32 %b, i32 %c, i32 %d) #0
  ret i32 %call1
}

; CHECK1-LABEL: f
; ...
; CHECK1:       aut r12, lr, sp
; CHECK1-NOT:   bx r12

; CHECK2-LABEL: f
; ...
; CHECK2:       blx r4
; CHECK2-NEXT:  ldr r12, [sp], #4
; CHECK2-NEXT:  pop.w {r4, r5, r6, r7, lr}
; CHECK2-NEXT:  aut r12, lr, sp
; CHECK2-NEXT:  bx lr

declare dso_local i32 @g(i32) local_unnamed_addr #0

attributes #0 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}
