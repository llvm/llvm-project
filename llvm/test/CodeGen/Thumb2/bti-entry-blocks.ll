; RUN: llc < %s -mtriple=thumbv7m-arm-none-eabi | FileCheck %s

define hidden i32 @linkage_external() local_unnamed_addr {
; CHECK-LABEL: linkage_external:
; CHECK: bti
; CHECK-NEXT: movs r0, #1
; CHECK-NEXT: bx lr
entry:
  ret i32 1
}

define internal i32 @linkage_internal() unnamed_addr {
; CHECK-LABEL: linkage_internal:
; CHECK: bti
; CHECK: movs r0, #2
; CHECK-NEXT: bx lr
entry:
  ret i32 2
}

!llvm.module.flags = !{!1}
!1 = !{i32 1, !"branch-target-enforcement", i32 1}
