; RUN: opt -S %s -lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/exported-funcs.yaml | FileCheck %s
;
; CHECK: module asm ".symver external_addrtaken, alias1"
; CHECK-NOT: .symver external_addrtaken2
; CHECK-NOT: .symver not_exported

target triple = "x86_64-unknown-linux"

!cfi.functions = !{!0, !1}
!symvers = !{!3, !4}

!0 = !{!"external_addrtaken", i8 0, !2}
!1 = !{!"external_addrtaken2", i8 0, !2}
!2 = !{i64 0, !"typeid1"}
!3 = !{!"external_addrtaken", !"alias1"}
!4 = !{!"not_exported", !"alias2"}
