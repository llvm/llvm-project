; RUN: llc -verify-machineinstrs -mtriple=aarch64-apple-ios -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT %s
; RUN: llc -O0 -fast-isel -verify-machineinstrs -mtriple=aarch64-apple-ios -o - %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=aarch64-unknown-linux-gnu -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT %s

; Parameter with swiftself should be allocated to x20.
; CHECK-LABEL: swiftself_param:
; CHECK: mov x0, x20
; CHECK-NEXT: ret
define i8* @swiftself_param(i8* swiftself %addr0) {
  ret i8 *%addr0
}

; Check that x20 is used to pass a swiftself argument.
; CHECK-LABEL: call_swiftself:
; CHECK: mov x20, x0
; CHECK: bl {{_?}}swiftself_param
; CHECK: ret
define i8 *@call_swiftself(i8* %arg) {
  %res = call i8 *@swiftself_param(i8* swiftself %arg)
  ret i8 *%res
}

; x20 should be saved by the callee even if used for swiftself
; CHECK-LABEL: swiftself_clobber:
; CHECK: {{stp|str}} {{.*}}x20{{.*}}sp
; ...
; CHECK: {{ldp|ldr}} {{.*}}x20{{.*}}sp
; CHECK: ret
define i8 *@swiftself_clobber(i8* swiftself %addr0) {
  call void asm sideeffect "", "~{x20}"()
  ret i8 *%addr0
}

; Demonstrate that we do not need any movs when calling multiple functions
; with swiftself argument.
; CHECK-LABEL: swiftself_passthrough:
; OPT-NOT: mov{{.*}}x20
; OPT: bl {{_?}}swiftself_param
; OPT-NOT: mov{{.*}}x20
; OPT-NEXT: bl {{_?}}swiftself_param
; OPT: ret
define void @swiftself_passthrough(i8* swiftself %addr0) {
  call i8 *@swiftself_param(i8* swiftself %addr0)
  call i8 *@swiftself_param(i8* swiftself %addr0)
  ret void
}

; We can use a tail call if the callee swiftself is the same as the caller one.
; CHECK-LABEL: swiftself_tail:
; OPT: b {{_?}}swiftself_param
; OPT-NOT: ret
define i8* @swiftself_tail(i8* swiftself %addr0) {
  call void asm sideeffect "", "~{x20}"()
  %res = tail call i8* @swiftself_param(i8* swiftself %addr0)
  ret i8* %res
}

; We can not use a tail call if the callee swiftself is not the same as the
; caller one.
; CHECK-LABEL: swiftself_notail:
; CHECK: mov x20, x0
; CHECK: bl {{_?}}swiftself_param
; CHECK: ret
define i8* @swiftself_notail(i8* swiftself %addr0, i8* %addr1) nounwind {
  %res = tail call i8* @swiftself_param(i8* swiftself %addr1)
  ret i8* %res
}

; We cannot pretend that 'x0' is alive across the thisreturn_attribute call as
; we normally would. We marked the first parameter with swiftself which means it
; will no longer be passed in x0.
declare swiftcc i8* @thisreturn_attribute(i8* returned swiftself)
; OPT-LABEL: swiftself_nothisreturn:
; OPT-DAG: ldr  x20, [x20]
; OPT-DAG: mov [[CSREG:x[1-9].*]], x8
; OPT: bl {{_?}}thisreturn_attribute
; OPT: str x0, {{\[}}[[CSREG]]
; OPT: ret
define hidden swiftcc void @swiftself_nothisreturn(i8** noalias nocapture sret, i8** noalias nocapture readonly swiftself) {
entry:
  %2 = load i8*, i8** %1, align 8
  %3 = tail call swiftcc i8* @thisreturn_attribute(i8* swiftself %2)
  store i8* %3, i8** %0, align 8
  ret void
}
