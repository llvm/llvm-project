; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: It should have at least one range!
; CHECK-NEXT: !0 = !{}
define i64 @noalias_addrspace__empty(ptr %ptr, i64 %val) {
  %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, !noalias.addrspace !0
  ret i64 %ret
}

; CHECK: Unfinished range!
; CHECK-NEXT: !1 = !{i32 0}
define i64 @noalias_addrspace__single_field(ptr %ptr, i64 %val) {
  %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, !noalias.addrspace !1
  ret i64 %ret
}

; CHECK: Range must not be empty!
; CHECK-NEXT: !2 = !{i32 0, i32 0}
define i64 @noalias_addrspace__0_0(ptr %ptr, i64 %val) {
  %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, !noalias.addrspace !2
  ret i64 %ret
}

; CHECK: noalias.addrspace type must be i32!
; CHECK-NEXT: %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, align 8, !noalias.addrspace !3
define i64 @noalias_addrspace__i64(ptr %ptr, i64 %val) {
  %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, !noalias.addrspace !3
  ret i64 %ret
}

; CHECK: The lower limit must be an integer!
define i64 @noalias_addrspace__fp(ptr %ptr, i64 %val) {
  %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, !noalias.addrspace !4
  ret i64 %ret
}

; CHECK: The lower limit must be an integer!
define i64 @noalias_addrspace__ptr(ptr %ptr, i64 %val) {
  %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, !noalias.addrspace !5
  ret i64 %ret
}

; CHECK: The lower limit must be an integer!
define i64 @noalias_addrspace__nonconstant(ptr %ptr, i64 %val) {
  %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, !noalias.addrspace !6
  ret i64 %ret
}

@gv0 = global i32 0
@gv1 = global i32 1

!0 = !{}
!1 = !{i32 0}
!2 = !{i32 0, i32 0}
!3 = !{i64 1, i64 5}
!4 = !{float 0.0, float 2.0}
!5 = !{ptr null, ptr addrspace(1) null}
!6 = !{i32 ptrtoint (ptr @gv0 to i32), i32 ptrtoint (ptr @gv1 to i32) }


