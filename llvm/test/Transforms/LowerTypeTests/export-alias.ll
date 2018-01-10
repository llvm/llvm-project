; RUN: opt -S %s -lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/use-typeid1-typeid2.yaml | FileCheck %s
;
; CHECK: @alias1 = weak alias void (), void ()* @f
; CHECK: @alias2 = hidden alias void (), void ()* @f
; CHECK: declare !type !1 void @alias3()
; CHECK-NOT: @alias3 = alias

target triple = "x86_64-unknown-linux"

!cfi.functions = !{!0, !2, !3}
!aliases = !{!4, !5, !6}

!0 = !{!"f", i8 0, !1}
!1 = !{i64 0, !"typeid1"}
!2 = !{!"alias1", i8 1, !1}
; alias2 not included here, this could happen if the only reference to alias2
; is in a module compiled without cfi-icall
!3 = !{!"alias3", i8 1, !1}
!4 = !{!"alias1", !"f", i8 0, i8 1}
!5 = !{!"alias2", !"f", i8 1, i8 0}
!6 = !{!"alias3", !"not_present", i8 0, i8 0}
