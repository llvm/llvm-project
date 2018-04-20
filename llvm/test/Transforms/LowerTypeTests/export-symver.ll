; RUN: opt -S %s -lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/use-typeid1-typeid2.yaml | FileCheck %s
;
; CHECK: module asm ".symver exported_and_symver, alias1"
; CHECK-NOT: .symver exported
; CHECK-NOT: .symver symver

target triple = "x86_64-unknown-linux"

!cfi.functions = !{!0, !1}
!symvers = !{!3, !4}

!0 = !{!"exported_and_symver", i8 2, !2}
!1 = !{!"exported", i8 2, !2}
!2 = !{i64 0, !"typeid1"}
!3 = !{!"exported_and_symver", !"alias1"}
!4 = !{!"symver", !"alias2"}
