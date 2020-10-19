; RUN: llc < %s -mtriple=ve | FileCheck %s

; Function Attrs: nounwind
define void @brcond_then(i1 zeroext %0) {
; CHECK-LABEL: brcond_then:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    breq.w 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  br i1 %0, label %2, label %3

2:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %3

3:                                                ; preds = %2, %1
  ret void
}

; Function Attrs: nounwind
define void @brcond_else(i1 zeroext %0) {
; CHECK-LABEL: brcond_else:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.w 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  br i1 %0, label %3, label %2

2:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %3

3:                                                ; preds = %2, %1
  ret void
}
