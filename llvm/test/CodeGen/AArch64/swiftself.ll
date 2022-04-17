; RUN: llc -verify-machineinstrs -mtriple=aarch64-apple-ios -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT --check-prefix=OPTAARCH64 %s
; RUN: llc -O0 -fast-isel -verify-machineinstrs -mtriple=aarch64-apple-ios -o - %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=aarch64-unknown-linux-gnu -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT --check-prefix=OPTAARCH64 %s
; RUN: llc -verify-machineinstrs -mtriple=arm64_32-apple-ios -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT --check-prefix=OPTARM64_32 %s

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
; This should also work with fast-isel.
; CHECK-LABEL: swiftself_tail:
; OPTAARCH64: b {{_?}}swiftself_param
; OPTAARCH64-NOT: ret
; OPTARM64_32: b {{_?}}swiftself_param
define i8* @swiftself_tail(i8* swiftself %addr0) {
  call void asm sideeffect "", "~{x20}"()
  %res = musttail call i8* @swiftself_param(i8* swiftself %addr0)
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
; OPTAARCH64-LABEL: swiftself_nothisreturn:
; OPTAARCH64-DAG: ldr  x20, [x20]
; OPTAARCH64-DAG: mov [[CSREG:x[1-9].*]], x8
; OPTAARCH64: bl {{_?}}thisreturn_attribute
; OPTAARCH64: str x0, [[[CSREG]]
; OPTAARCH64: ret

; OPTARM64_32-LABEL: swiftself_nothisreturn:
; OPTARM64_32-DAG: ldr  w20, [x20]
; OPTARM64_32-DAG: mov [[CSREG:x[1-9].*]], x8
; OPTARM64_32: bl {{_?}}thisreturn_attribute
; OPTARM64_32: str w0, [[[CSREG]]
; OPTARM64_32: ret
define hidden swiftcc void @swiftself_nothisreturn(i8** noalias nocapture sret(i8*), i8** noalias nocapture readonly swiftself) {
entry:
  %2 = load i8*, i8** %1, align 8
  %3 = tail call swiftcc i8* @thisreturn_attribute(i8* swiftself %2)
  store i8* %3, i8** %0, align 8
  ret void
}
