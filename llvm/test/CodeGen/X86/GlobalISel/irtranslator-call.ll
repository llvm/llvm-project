; RUN: llc -mtriple i386 -global-isel -stop-after=irtranslator %s -o - | FileCheck %s
; RUN: llc -mtriple x86_64 -global-isel -stop-after=irtranslator %s -o - | FileCheck %s

define void @test_void_return() {
; CHECK-LABEL: name:            test_void_return
; CHECK:      alignment:       4
; CHECK-NEXT: exposesReturnsTwice: false
; CHECK-NEXT: legalized:       false
; CHECK-NEXT: regBankSelected: false
; CHECK-NEXT: selected:        false
; CHECK-NEXT: tracksRegLiveness: true
; CHECK-NEXT: frameInfo:
; CHECK-NEXT:   isFrameAddressTaken: false
; CHECK-NEXT:   isReturnAddressTaken: false
; CHECK-NEXT:   hasStackMap:     false
; CHECK-NEXT:   hasPatchPoint:   false
; CHECK-NEXT:   stackSize:       0
; CHECK-NEXT:   offsetAdjustment: 0
; CHECK-NEXT:   maxAlignment:    0
; CHECK-NEXT:   adjustsStack:    false
; CHECK-NEXT:   hasCalls:        false
; CHECK-NEXT:   maxCallFrameSize: 0
; CHECK-NEXT:   hasOpaqueSPAdjustment: false
; CHECK-NEXT:   hasVAStart:      false
; CHECK-NEXT:   hasMustTailInVarArgFunc: false
; CHECK-NEXT: body:
; CHECK-NEXT:   bb.1.entry:
; CHECK-NEXT:     RET 0
entry:
  ret void
}
