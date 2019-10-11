; RUN: opt -S %s -lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/exported-funcs.yaml | FileCheck %s

; CHECK: define internal void @external_addrtaken.1()
; CHECK: declare {{.*}} void @external_addrtaken.cfi()

target triple = "x86_64-unknown-linux"

define internal void @external_addrtaken() !type !1 {
  ret void
}

!cfi.functions = !{!0}

!0 = !{!"external_addrtaken", i8 0, !1}
!1 = !{i64 0, !"typeid1"}
