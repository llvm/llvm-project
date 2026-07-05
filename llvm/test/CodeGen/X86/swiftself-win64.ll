; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown-windows-msvc -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=x86_64-unknown-windows-msvc -o - %s | FileCheck %s

; Parameter with swiftself should be allocated to r13.
; CHECK-LABEL: swiftself_param:
; CHECK: movq %r13, %rax
define ptr@swiftself_param(ptr swiftself %addr0) {
    ret ptr%addr0
}

; Check that r13 is used to pass a swiftself argument.
; CHECK-LABEL: call_swiftself:
; CHECK: movq %rcx, %r13
; CHECK: callq swiftself_param
define ptr@call_swiftself(ptr %arg) {
  %res = call ptr@swiftself_param(ptr swiftself %arg)
  ret ptr%res
}

; r13 should be saved by the callee even if used for swiftself
; CHECK-LABEL: swiftself_clobber:
; CHECK: pushq %r13
; ...
; CHECK: popq %r13
define ptr@swiftself_clobber(ptr swiftself %addr0) {
  call void asm sideeffect "nop", "~{r13}"()
  ret ptr%addr0
}

; Demonstrate that we do not need any movs when calling multiple functions
; with swiftself argument.
; CHECK-LABEL: swiftself_passthrough:
; OPT-NOT: mov{{.*}}r13
; OPT: callq swiftself_param
; OPT-NOT: mov{{.*}}r13
; OPT-NEXT: callq swiftself_param
define void @swiftself_passthrough(ptr swiftself %addr0) {
  call ptr@swiftself_param(ptr swiftself %addr0)
  call ptr@swiftself_param(ptr swiftself %addr0)
  ret void
}

; We can use a tail call if the callee swiftself is the same as the caller one.
; This should also work with fast-isel.
; CHECK-LABEL: swiftself_tail:
; CHECK: jmp swiftself_param
; CHECK-NOT: ret
define ptr @swiftself_tail(ptr swiftself %addr0) {
  call void asm sideeffect "", "~{r13}"()
  %res = tail call ptr @swiftself_param(ptr swiftself %addr0)
  ret ptr %res
}

; We can not use a tail call if the callee swiftself is not the same as the
; caller one.
; CHECK-LABEL: swiftself_notail:
; CHECK: movq %rcx, %r13
; CHECK: callq swiftself_param
; CHECK: retq
define ptr @swiftself_notail(ptr swiftself %addr0, ptr %addr1) nounwind {
  %res = tail call ptr @swiftself_param(ptr swiftself %addr1)
  ret ptr %res
}
