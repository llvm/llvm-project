; RUN: opt %s -disable-output -debugify-each -debugify-quiet -debugify-export - -passes=globalopt | FileCheck %s
; RUN: opt %s -disable-output -debugify-each -debugify-quiet -debugify-export - -passes=globalopt | FileCheck %s

; RUN: opt --experimental-debuginfo-iterators=false %s -disable-output -debugify-each -debugify-quiet -debugify-export - -passes=globalopt | FileCheck %s
; RUN: opt --experimental-debuginfo-iterators=false %s -disable-output -debugify-each -debugify-quiet -debugify-export - -passes=globalopt | FileCheck %s

; CHECK: Pass Name
; CHECK-SAME: # of missing debug values
; CHECK-SAME: # of missing locations
; CHECK-SAME: Missing/Expected value ratio
; CHECK-SAME: Missing/Expected location ratio

; CHECK:      {{Module Verifier|GlobalOptPass}}
; CHECK-SAME: 0,0,0.000000e+00,0.000000e+00

define void @foo() {
  ret void
}
