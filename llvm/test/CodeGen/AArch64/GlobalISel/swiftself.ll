; RUN: llc -verify-machineinstrs -mtriple=aarch64-apple-ios -o - %s -global-isel | FileCheck %s

; Parameter with swiftself should be allocated to x20.
; CHECK-LABEL: swiftself_param:
; CHECK: mov x0, x20
; CHECK-NEXT: ret
define ptr @swiftself_param(ptr swiftself %addr0) {
  ret ptr %addr0
}

; Check that x20 is used to pass a swiftself argument.
; CHECK-LABEL: call_swiftself:
; CHECK: mov x20, x0
; CHECK: bl {{_?}}swiftself_param
; CHECK: ret
define ptr @call_swiftself(ptr %arg) {
  %res = call ptr @swiftself_param(ptr swiftself %arg)
  ret ptr %res
}

; Demonstrate that we do not need any movs when calling multiple functions
; with swiftself argument.
; CHECK-LABEL: swiftself_passthrough:
; CHECK-NOT: mov{{.*}}x20
; CHECK: bl {{_?}}swiftself_param
; CHECK-NOT: mov{{.*}}x20
; CHECK-NEXT: bl {{_?}}swiftself_param
; CHECK: ret
define void @swiftself_passthrough(ptr swiftself %addr0) {
  call ptr @swiftself_param(ptr swiftself %addr0)
  call ptr @swiftself_param(ptr swiftself %addr0)
  ret void
}

; We can not use a tail call if the callee swiftself is not the same as the
; caller one.
; CHECK-LABEL: swiftself_notail:
; CHECK: mov x20, x0
; CHECK: bl {{_?}}swiftself_param
; CHECK: ret
define ptr @swiftself_notail(ptr swiftself %addr0, ptr %addr1) nounwind {
  %res = tail call ptr @swiftself_param(ptr swiftself %addr1)
  ret ptr %res
}

; We cannot pretend that 'x0' is alive across the thisreturn_attribute call as
; we normally would. We marked the first parameter with swiftself which means it
; will no longer be passed in x0.
declare swiftcc ptr @thisreturn_attribute(ptr returned swiftself)
; CHECK-LABEL: swiftself_nothisreturn:
; CHECK-DAG: ldr  x20, [x20]
; CHECK-DAG: mov [[CSREG:x[1-9].*]], x8
; CHECK: bl {{_?}}thisreturn_attribute
; CHECK: str x0, [[[CSREG]]
; CHECK: ret
define hidden swiftcc void @swiftself_nothisreturn(ptr noalias nocapture sret(ptr), ptr noalias nocapture readonly swiftself) {
entry:
  %2 = load ptr, ptr %1, align 8
  %3 = tail call swiftcc ptr @thisreturn_attribute(ptr swiftself %2)
  store ptr %3, ptr %0, align 8
  ret void
}

; Check that x20 is used to pass a swiftself argument when the parameter is
; only in the declaration's arguments.
; CHECK-LABEL: _swiftself_not_on_call_params:
; CHECK: mov x20, x0
; CHECK: bl {{_?}}swiftself_param
; CHECK: ret
define ptr @swiftself_not_on_call_params(ptr %arg) {
  %res = call ptr @swiftself_param(ptr %arg)
  ret ptr %res
}
