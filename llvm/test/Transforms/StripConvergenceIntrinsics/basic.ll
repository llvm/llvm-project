; RUN: opt -passes=strip-convergence-intrinsics -S < %s | FileCheck %s

; Verify that convergence intrinsics and convergencectrl operand bundles are
; stripped by the pass.

define void @entry_and_call() convergent {
; CHECK-LABEL: define void @entry_and_call()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @convergent_callee()
; CHECK-NOT:     convergence
; CHECK-NEXT:    ret void
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  call void @convergent_callee() [ "convergencectrl"(token %0) ]
  ret void
}

define void @loop_with_token() convergent {
; CHECK-LABEL: define void @loop_with_token()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    call void @convergent_callee()
; CHECK-NOT:     convergence
; CHECK-NEXT:    br i1
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  br label %loop

loop:
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  call void @convergent_callee() [ "convergencectrl"(token %1) ]
  br i1 true, label %exit, label %loop

exit:
  ret void
}

define void @anchor_token() convergent {
; CHECK-LABEL: define void @anchor_token()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @convergent_callee()
; CHECK-NOT:     convergence
; CHECK-NEXT:    ret void
entry:
  %0 = call token @llvm.experimental.convergence.anchor()
  call void @convergent_callee() [ "convergencectrl"(token %0) ]
  ret void
}

; This ensures that when the use of a token is defined before the call that
; creates the token, we are still able to strip them as expected.
define void @reversed_block_order() convergent {
; CHECK-LABEL: define void @reversed_block_order()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %[[B:.*]]
; CHECK:       [[A:.*]]:
; CHECK-NEXT:    call void @convergent_callee()
; CHECK-NEXT:    ret void
; CHECK:      [[B]]:
; CHECK-NEXT:    br label %[[A]]
entry:
 br label %B
A:
 call void @convergent_callee() [ "convergencectrl"(token %tok) ]
 ret void
B:
 %tok = call token @llvm.experimental.convergence.anchor()
 br label %A
}

; Declarations may remain but should have no uses in function bodies.
; CHECK-LABEL: declare void @convergent_callee

declare void @convergent_callee() convergent

declare token @llvm.experimental.convergence.entry()
declare token @llvm.experimental.convergence.loop()
declare token @llvm.experimental.convergence.anchor()
