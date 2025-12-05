; RUN: llc -mtriple=aarch64--linux-gnu < %s | FileCheck %s

define dso_local void @_Z3fooi(ptr %x, ptr %y) #0 {
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
  %0 = call <8 x i32> asm sideeffect "", "={z1}"()
  store <8 x i32> %0, ptr %y, align 16
  %13 = load <8 x i32>, ptr %x, align 16
  call void asm sideeffect "", "{z0}"(<8 x i32> %13)
  ret void
}
