; RUN: llc -mtriple thumbv8.1m.main-arm-none-eabi %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-USE
; RUN: llc -mtriple thumbv8.1m.main-arm-none-eabi -mattr=+pacbti %s -o - | FileCheck %s --check-prefixes=CHECK-ARCHEXT,CHECK-USE

; CHECK-DAG:         .eabi_attribute	50, 1	@ Tag_PAC_extension
; CHECK-ARCHEXT-DAG: .eabi_attribute	50, 2	@ Tag_PAC_extension
; CHECK-DAG:         .eabi_attribute	52, 1	@ Tag_BTI_extension
; CHECK-ARCHEXT-DAG: .eabi_attribute	52, 2	@ Tag_BTI_extension
; CHECK-USE-DAG:     .eabi_attribute	76, 1	@ Tag_PACRET_use
; CHECK-USE-DAG:     .eabi_attribute	74, 1	@ Tag_BTI_use

define i32 @foo(i32 %a) {
entry:
  %add = add nsw i32 %a, 1
  ret i32 %add
}

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"branch-target-enforcement", i32 1}
!1 = !{i32 1, !"sign-return-address", i32 1}
!2 = !{i32 1, !"sign-return-address-all", i32 0}
