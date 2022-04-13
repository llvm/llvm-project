;; RUN: llc -mtriple=aarch64-eabi -mattr=+v8.5a %s -o - | FileCheck %s

declare i32 @g(i32) #5

define i32 @f0(i32 %x) #0 {
entry:
  %call = tail call i32 @g(i32 %x) #5
  %add = add nsw i32 %call, 1
  ret i32 %add
}
;; CHECK-LABEL: f0:
;; CHECK-NOT:   bti
;; CHECK-NOT:   pacia
;; CHECK-NOT:   reta

define i32 @f1(i32 %x) #1 {
entry:
  %call = tail call i32 @g(i32 %x) #5
  %add = add nsw i32 %call, 1
  ret i32 %add
}
;; CHECK-LABEL: f1:
;; CHECK:       bti c
;; CHECK-NOT:   reta

define i32 @f2(i32 %x) #2 {
entry:
  %call = tail call i32 @g(i32 %x) #5
  %add = add nsw i32 %call, 1
  ret i32 %add
}
;; CHECK-LABEL: f2:
;; CHECK:       pacia x30, sp
;; CHECK:       retaa

define i32 @f3(i32 %x) #3 {
entry:
  %call = tail call i32 @g(i32 %x) #5
  %add = add nsw i32 %call, 1
  ret i32 %add
}
;; CHECK-LABEL: f3:
;; CHECK:       pacib x30, sp
;; CHECK:       retab

define i32 @f4(i32 %x) #4 {
entry:
  ret i32 1
}
;; CHECK-LABEL: f4:
;; CHECK:       pacia x30, sp
;; CHECK:       retaa

define i32 @f5(i32 %x) #5 {
entry:
  %call = tail call i32 @g(i32 %x) #5
  %add = add nsw i32 %call, 1
  ret i32 %add
}
;; CHECK-LABEL: f5:
;; CHECK:       pacia x30, sp
;; CHECK:       retaa

attributes #0 = { nounwind "branch-target-enforcement"="false" "sign-return-address"="none" }
attributes #1 = { nounwind "branch-target-enforcement"="true"  "sign-return-address"="none" }
attributes #2 = { nounwind "branch-target-enforcement"="false" "sign-return-address"="non-leaf" "sign-return-address-key"="a_key" }
attributes #3 = { nounwind "branch-target-enforcement"="false" "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" }
attributes #4 = { nounwind "branch-target-enforcement"="false" "sign-return-address"="all" "sign-return-address-key"="a_key" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"branch-target-enforcement", i32 1}
!2 = !{i32 8, !"sign-return-address", i32 1}
!3 = !{i32 8, !"sign-return-address-all", i32 0}
!4 = !{i32 8, !"sign-return-address-with-bkey", i32 0}
