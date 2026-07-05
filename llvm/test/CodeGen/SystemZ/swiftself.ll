; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Parameter with swiftself should be allocated to r10.
; CHECK-LABEL: swiftself_param:
; CHECK: lgr %r2, %r10
define ptr@swiftself_param(ptr swiftself %addr0) {
  ret ptr %addr0
}

; Check that r10 is used to pass a swiftself argument.
; CHECK-LABEL: call_swiftself:
; CHECK: lgr %r10, %r2
; CHECK: brasl %r14, swiftself_param
define ptr@call_swiftself(ptr %arg) {
  %res = call ptr@swiftself_param(ptr swiftself %arg)
  ret ptr %res
}

; r10 should be saved by the callee even if used for swiftself
; CHECK-LABEL: swiftself_clobber:
; CHECK: stmg %r10,
; ...
; CHECK: lmg %r10,
; CHECK: br %r14
define ptr@swiftself_clobber(ptr swiftself %addr0) {
  call void asm sideeffect "", "~{r10}"()
  ret ptr %addr0
}

; Demonstrate that we do not need any loads when calling multiple functions
; with swiftself argument.
; CHECK-LABEL: swiftself_passthrough:
; CHECK-NOT: lg{{.*}}r10,
; CHECK: brasl %r14, swiftself_param
; CHECK-NOT: lg{{.*}}r10,
; CHECK-NEXT: brasl %r14, swiftself_param
define void @swiftself_passthrough(ptr swiftself %addr0) {
  call ptr@swiftself_param(ptr swiftself %addr0)
  call ptr@swiftself_param(ptr swiftself %addr0)
  ret void
}

; Normally, we can use a tail call if the callee swiftself is the same as the
; caller one. Not yet supported on SystemZ.
; CHECK-LABEL: swiftself_tail:
; CHECK: lgr %r[[REG1:[0-9]+]], %r10
; CHECK: lgr %r10, %r[[REG1]]
; CHECK: brasl %r14, swiftself_param
; CHECK: br %r14
define ptr @swiftself_tail(ptr swiftself %addr0) {
  call void asm sideeffect "", "~{r10}"()
  %res = tail call ptr @swiftself_param(ptr swiftself %addr0)
  ret ptr %res
}

; We can not use a tail call if the callee swiftself is not the same as the
; caller one.
; CHECK-LABEL: swiftself_notail:
; CHECK: lgr %r10, %r2
; CHECK: brasl %r14, swiftself_param
; CHECK: lmg %r10,
; CHECK: br %r14
define ptr @swiftself_notail(ptr swiftself %addr0, ptr %addr1) nounwind {
  %res = tail call ptr @swiftself_param(ptr swiftself %addr1)
  ret ptr %res
}
