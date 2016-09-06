; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Parameter with swiftself should be allocated to r10.
; CHECK-LABEL: swiftself_param:
; CHECK: lgr %r2, %r10
define i8 *@swiftself_param(i8* swiftself %addr0) {
  ret i8 *%addr0
}

; Check that r10 is used to pass a swiftself argument.
; CHECK-LABEL: call_swiftself:
; CHECK: lgr %r10, %r2
; CHECK: brasl %r14, swiftself_param
define i8 *@call_swiftself(i8* %arg) {
  %res = call i8 *@swiftself_param(i8* swiftself %arg)
  ret i8 *%res
}

; r10 should be saved by the callee even if used for swiftself
; CHECK-LABEL: swiftself_clobber:
; CHECK: stmg %r10,
; ...
; CHECK: lmg %r10,
; CHECK: br %r14
define i8 *@swiftself_clobber(i8* swiftself %addr0) {
  call void asm sideeffect "", "~{r10}"()
  ret i8 *%addr0
}

; Demonstrate that we do not need any loads when calling multiple functions
; with swiftself argument.
; CHECK-LABEL: swiftself_passthrough:
; CHECK-NOT: lg{{.*}}r10,
; CHECK: brasl %r14, swiftself_param
; CHECK-NOT: lg{{.*}}r10,
; CHECK-NEXT: brasl %r14, swiftself_param
define void @swiftself_passthrough(i8* swiftself %addr0) {
  call i8 *@swiftself_param(i8* swiftself %addr0)
  call i8 *@swiftself_param(i8* swiftself %addr0)
  ret void
}

; Normally, we can use a tail call if the callee swiftself is the same as the
; caller one. Not yet supported on SystemZ.
; CHECK-LABEL: swiftself_tail:
; CHECK: lgr %r[[REG1:[0-9]+]], %r10
; CHECK: lgr %r10, %r[[REG1]]
; CHECK: brasl %r14, swiftself_param
; CHECK: br %r14
define i8* @swiftself_tail(i8* swiftself %addr0) {
  call void asm sideeffect "", "~{r10}"()
  %res = tail call i8* @swiftself_param(i8* swiftself %addr0)
  ret i8* %res
}

; We can not use a tail call if the callee swiftself is not the same as the
; caller one.
; CHECK-LABEL: swiftself_notail:
; CHECK: lgr %r10, %r2
; CHECK: brasl %r14, swiftself_param
; CHECK: lmg %r10,
; CHECK: br %r14
define i8* @swiftself_notail(i8* swiftself %addr0, i8* %addr1) nounwind {
  %res = tail call i8* @swiftself_param(i8* swiftself %addr1)
  ret i8* %res
}
