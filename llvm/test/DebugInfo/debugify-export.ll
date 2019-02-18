; RUN: opt < %s -debugify-each -debugify-quiet -debugify-export - -o /dev/null | FileCheck %s

; CHECK: Pass Name
; CHECK-SAME: # of missing debug values
; CHECK-SAME: # of missing locations
; CHECK-SAME: Missing/Expected value ratio
; CHECK-SAME: Missing/Expected location ratio

; CHECK: Module Verifier
; CHECK-SAME: 0,0,0.000000e+00,0.000000e+00

define void @foo() {
  ret void
}
