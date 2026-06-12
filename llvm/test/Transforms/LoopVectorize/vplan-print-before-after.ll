; RUN: opt -passes=loop-vectorize -disable-output  -force-vector-width=4  < %s \
; RUN:   -vplan-print-after=simplify -vplan-print-after=printFinalVPlan \
; RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-AFTER --implicit-check-not "VPlan after"
; RUN: opt -passes=loop-vectorize -disable-output  -force-vector-width=4  < %s \
; RUN:   -vplan-print-before=simplify -vplan-print-before=printFinalVPlan \
; RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-BEFORE --implicit-check-not "VPlan after"
; REQUIRES: asserts

; CHECK-BEFORE:      VPlan for loop in 'foo' before simplifyRecipes
; CHECK-BEFORE-NEXT: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-BEFORE:      VPlan for loop in 'foo' before simplifyBlends
; CHECK-BEFORE-NEXT: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-BEFORE:      VPlan for loop in 'foo' before simplifyRecipes
; CHECK-BEFORE-NEXT: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-BEFORE:      VPlan for loop in 'foo' before printFinalVPlan
; CHECK-BEFORE-NEXT: VPlan 'Final VPlan for VF={4},UF={1}' {

; CHECK-AFTER:      VPlan for loop in 'foo' after simplifyRecipes
; CHECK-AFTER-NEXT: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-AFTER:      VPlan for loop in 'foo' after simplifyBlends
; CHECK-AFTER-NEXT: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-AFTER:      VPlan for loop in 'foo' after simplifyRecipes
; CHECK-AFTER-NEXT: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-AFTER:      VPlan for loop in 'foo' after printFinalVPlan
; CHECK-AFTER-NEXT: VPlan 'Final VPlan for VF={4},UF={1}' {

define void @foo(ptr %ptr, i64 %n) {
entry:
  br label %header

header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %header ]
  %gep = getelementptr i64, ptr %ptr, i64 %iv
  store i64 %iv, ptr %gep
  %iv.next = add nsw i64 %iv, 1
  %exitcond = icmp slt i64 %iv.next, %n
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}
