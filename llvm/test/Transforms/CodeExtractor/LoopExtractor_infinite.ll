; RUN: opt < %s -mergereturn -loop-extract -enable-new-pm=0 -S | FileCheck %s

; This test used to enter an infinite loop, until out of memory (PR3082).

define void @test() {
; CHECK-LABEL: define void @test()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %codeRepl
; CHECK:       codeRepl:
; CHECK-NEXT:    %targetBlock = call i1 @test.loopentry()
; CHECK-NEXT:    br i1 %targetBlock, label %exit.1, label %exit.0
; CHECK:       exit.0:
; CHECK-NEXT:    br label %UnifiedReturnBlock
; CHECK:       exit.1:
; CHECK-NEXT:    br label %UnifiedReturnBlock
; CHECK:       UnifiedReturnBlock:
; CHECK-NEXT:    ret void

entry:
  br label %loopentry

loopentry:                                        ; preds = %loopexit, %entry
  br i1 undef, label %exit.1, label %loopexit

loopexit:                                         ; preds = %loopentry
  br i1 undef, label %loopentry, label %exit.0

exit.0:                                           ; preds = %loopexit
  ret void

exit.1:                                           ; preds = %loopentry
  ret void
}

; CHECK-LABEL: define internal i1 @test.loopentry()
; CHECK-NEXT:  newFuncRoot:
; CHECK-NEXT:    br label %loopentry
; CHECK:       exit.1.exitStub:
; CHECK-NEXT:    ret i1 true
; CHECK:       exit.0.exitStub:
; CHECK-NEXT:    ret i1 false
; CHECK:       loopentry:
; CHECK-NEXT:    br i1 true, label %exit.1.exitStub, label %loopexit
; CHECK:       loopexit:
; CHECK-NEXT:    br i1 false, label %loopexit.loopentry_crit_edge, label %exit.0.exitStub
; CHECK:       loopexit.loopentry_crit_edge:
; CHECK-NEXT:    br label %loopentry
