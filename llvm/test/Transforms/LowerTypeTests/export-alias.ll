; RUN: opt -S %s -passes=lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/exported-funcs.yaml | FileCheck %s
;
; CHECK: @alias1 = alias [8 x i8], ptr @external_addrtaken
; CHECK: @alias2 = alias [8 x i8], ptr @external_addrtaken
; CHECK-NOT: @alias3 = alias
; CHECK-NOT: @not_present

target triple = "x86_64-unknown-linux"

!cfi.functions = !{!0, !2, !3, !4}
!aliases = !{!5, !6}

!0 = !{!"external_addrtaken", i8 0, !1}
!1 = !{i64 0, !"typeid1"}
!2 = !{!"alias1", i8 0, !1}
!3 = !{!"alias2", i8 0, !1}
!4 = !{!"alias3", i8 0, !1}
!5 = !{!"external_addrtaken", !"alias1", !"alias2"}
!6 = !{!"not_present", !"alias3"}
