; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown-unknown -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=x86_64-unknown-unknown -o - %s | FileCheck %s

; Parameter with swiftself should be allocated to r13.
; CHECK-LABEL: swiftself_param:
; CHECK: movq %r13, %rax
define i8 *@swiftself_param(i8* swiftself %addr0) {
    ret i8 *%addr0
}

; Check that r13 is used to pass a swiftself argument.
; CHECK-LABEL: call_swiftself:
; CHECK: movq %rdi, %r13
; CHECK: callq {{_?}}swiftself_param
define i8 *@call_swiftself(i8* %arg) {
  %res = call i8 *@swiftself_param(i8* swiftself %arg)
  ret i8 *%res
}

; r13 should be saved by the callee even if used for swiftself
; CHECK-LABEL: swiftself_clobber:
; CHECK: pushq %r13
; ...
; CHECK: popq %r13
define i8 *@swiftself_clobber(i8* swiftself %addr0) {
  call void asm sideeffect "nop", "~{r13}"()
  ret i8 *%addr0
}

; Demonstrate that we do not need any movs when calling multiple functions
; with swiftself argument.
; CHECK-LABEL: swiftself_passthrough:
; OPT-NOT: mov{{.*}}r13
; OPT: callq {{_?}}swiftself_param
; OPT-NOT: mov{{.*}}r13
; OPT-NEXT: callq {{_?}}swiftself_param
define void @swiftself_passthrough(i8* swiftself %addr0) {
  call i8 *@swiftself_param(i8* swiftself %addr0)
  call i8 *@swiftself_param(i8* swiftself %addr0)
  ret void
}

; We can use a tail call if the callee swiftself is the same as the caller one.
; This should also work with fast-isel.
; CHECK-LABEL: swiftself_tail:
; CHECK: jmp {{_?}}swiftself_param
; CHECK-NOT: ret
define i8* @swiftself_tail(i8* swiftself %addr0) {
  call void asm sideeffect "", "~{r13}"()
  %res = tail call i8* @swiftself_param(i8* swiftself %addr0)
  ret i8* %res
}

; We can not use a tail call if the callee swiftself is not the same as the
; caller one.
; CHECK-LABEL: swiftself_notail:
; CHECK: movq %rdi, %r13
; CHECK: callq {{_?}}swiftself_param
; CHECK: retq
define i8* @swiftself_notail(i8* swiftself %addr0, i8* %addr1) nounwind {
  %res = tail call i8* @swiftself_param(i8* swiftself %addr1)
  ret i8* %res
}
