; RUN: opt < %s -passes='cgscc(inline,loop-simplify),loop-extract' -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(argpromotion,loop-simplify),loop-extract' -S | FileCheck %s

; This test used to trigger an assert (PR8929).

define void @test() {
; CHECK-LABEL: define void @test()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %codeRepl
; CHECK:       codeRepl:
; CHECK-NEXT:    call void @test.loopentry()
; CHECK-NEXT:    br label %loopexit
; CHECK:       loopexit:
; CHECK-NEXT:    br label %exit
; CHECK:       exit:
; CHECK-NEXT:    ret void

entry:
  br label %loopentry

loopentry:                                        ; preds = %loopbody, %entry
  br i1 undef, label %loopbody, label %loopexit

loopbody:                                         ; preds = %codeRepl1
  call void @foo()
  br label %loopentry

loopexit:                                         ; preds = %codeRepl
  br label %exit

exit:                                             ; preds = %loopexit
  ret void
}

declare void @foo()

; CHECK-LABEL: define internal void @test.loopentry()
; CHECK-NEXT:  newFuncRoot:
; CHECK-NEXT:    br label %loopentry
; CHECK:       loopentry:
; CHECK-NEXT:    br i1 false, label %loopbody, label %loopexit.exitStub
; CHECK:       loopbody:
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    br label %loopentry
; CHECK:       loopexit.exitStub:
; CHECK-NEXT:    ret void
