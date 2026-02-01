; RUN: llc -mtriple=aarch64--linux < %s | FileCheck %s

define void @_Z3fooi(ptr %x, ptr %y) #0 {
; CHECK-LABEL: _Z3fooi:
; CHECK:       // %bb.0:
; CHECK-NEXT:    //APP
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    stp	q1, q2, [x1]
; CHECK-NEXT:    ldp	q0, q1, [x0]
; CHECK-NEXT:    //APP
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    ret
entry:
  %y.y = call <8 x i32> asm sideeffect "", "={z1}"()
  store <8 x i32> %y.y, ptr %y, align 16
  %x.x = load <8 x i32>, ptr %x, align 16
  call void asm sideeffect "", "{z0}"(<8 x i32> %x.x)
  ret void
}
