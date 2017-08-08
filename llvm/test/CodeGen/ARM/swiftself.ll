; RUN: llc -verify-machineinstrs -mtriple=armv7k-apple-ios8.0 -mcpu=cortex-a7 -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT --check-prefix=TAILCALL %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=armv7k-apple-ios8.0 -mcpu=cortex-a7 -o - %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple=armv7-apple-ios -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=armv7-apple-ios -o - %s | FileCheck %s

; Parameter with swiftself should be allocated to r10.
; CHECK-LABEL: swiftself_param:
; CHECK: mov r0, r10
define i8 *@swiftself_param(i8* swiftself %addr0) "no-frame-pointer-elim"="true" {
    ret i8 *%addr0
}

; Check that r10 is used to pass a swiftself argument.
; CHECK-LABEL: call_swiftself:
; CHECK: mov r10, r0
; CHECK: bl {{_?}}swiftself_param
define i8 *@call_swiftself(i8* %arg) "no-frame-pointer-elim"="true" {
  %res = call i8 *@swiftself_param(i8* swiftself %arg)
  ret i8 *%res
}

; r10 should be saved by the callee even if used for swiftself
; CHECK-LABEL: swiftself_clobber:
; CHECK: push {r10}
; ...
; CHECK: pop {r10}
define i8 *@swiftself_clobber(i8* swiftself %addr0) "no-frame-pointer-elim"="true" {
  call void asm sideeffect "", "~{r10}"()
  ret i8 *%addr0
}

; Demonstrate that we do not need any movs when calling multiple functions
; with swiftself argument.
; CHECK-LABEL: swiftself_passthrough:
; OPT-NOT: mov{{.*}}r10
; OPT: bl {{_?}}swiftself_param
; OPT-NOT: mov{{.*}}r10
; OPT-NEXT: bl {{_?}}swiftself_param
define void @swiftself_passthrough(i8* swiftself %addr0) "no-frame-pointer-elim"="true" {
  call i8 *@swiftself_param(i8* swiftself %addr0)
  call i8 *@swiftself_param(i8* swiftself %addr0)
  ret void
}

; We can use a tail call if the callee swiftself is the same as the caller one.
; CHECK-LABEL: swiftself_tail:
; TAILCALL: b {{_?}}swiftself_param
; TAILCALL-NOT: pop
define i8* @swiftself_tail(i8* swiftself %addr0) "no-frame-pointer-elim"="true" {
  call void asm sideeffect "", "~{r10}"()
  %res = tail call i8* @swiftself_param(i8* swiftself %addr0)
  ret i8* %res
}

; We can not use a tail call if the callee swiftself is not the same as the
; caller one.
; CHECK-LABEL: swiftself_notail:
; CHECK: mov r10, r0
; CHECK: bl {{_?}}swiftself_param
; CHECK: pop
define i8* @swiftself_notail(i8* swiftself %addr0, i8* %addr1) nounwind "no-frame-pointer-elim"="true" {
  %res = tail call i8* @swiftself_param(i8* swiftself %addr1)
  ret i8* %res
}

; We cannot pretend that 'r0' is alive across the thisreturn_attribute call as
; we normally would. We marked the first parameter with swiftself which means it
; will no longer be passed in r0.
declare swiftcc i8* @thisreturn_attribute(i8* returned swiftself)
; OPT-LABEL: swiftself_nothisreturn:
; OPT-DAG: mov [[CSREG:r[1-9].*]], r0
; OPT-DAG: ldr r10, [r10]
; OPT: bl  {{_?}}thisreturn_attribute
; OPT: str r0, {{\[}}[[CSREG]]
define hidden swiftcc void @swiftself_nothisreturn(i8** noalias nocapture sret, i8** noalias nocapture readonly swiftself) {
entry:
  %2 = load i8*, i8** %1, align 8
  %3 = tail call swiftcc i8* @thisreturn_attribute(i8* swiftself %2)
  store i8* %3, i8** %0, align 8
  ret void
}
