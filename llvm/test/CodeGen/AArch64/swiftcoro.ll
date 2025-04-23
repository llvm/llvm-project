; RUN: llc -verify-machineinstrs -mtriple=aarch64-apple-ios -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT --check-prefix=OPTAARCH64 %s
; RUN: llc -O0 -fast-isel -verify-machineinstrs -mtriple=aarch64-apple-ios -o - %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=aarch64-unknown-linux-gnu -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT --check-prefix=OPTAARCH64 %s
; RUN: llc -verify-machineinstrs -mtriple=arm64_32-apple-ios -o - %s | FileCheck --check-prefix=CHECK --check-prefix=OPT --check-prefix=OPTARM64_32 %s

; Parameter with swiftcoro should be allocated to x23.
; CHECK-LABEL: swiftcoro_param:
; CHECK: mov x0, x23
; CHECK-NEXT: ret
define ptr @swiftcoro_param(ptr swiftcoro %addr0) {
  ret ptr %addr0
}

; Check that x23 is used to pass a swiftcoro argument.
; CHECK-LABEL: call_swiftcoro:
; CHECK: mov x23, x0
; CHECK: bl {{_?}}swiftcoro_param
; CHECK: ret
define ptr @call_swiftcoro(ptr %arg) {
  %res = call ptr @swiftcoro_param(ptr swiftcoro %arg)
  ret ptr %res
}

; x23 should be saved by the callee even if used for swiftcoro
; CHECK-LABEL: swiftcoro_clobber:
; CHECK: {{stp|str}} {{.*}}x23{{.*}}sp
; ...
; CHECK: {{ldp|ldr}} {{.*}}x23{{.*}}sp
; CHECK: ret
define ptr @swiftcoro_clobber(ptr swiftcoro %addr0) {
  call void asm sideeffect "", "~{x23}"()
  ret ptr %addr0
}

; Demonstrate that we do not need any movs when calling multiple functions
; with swiftcoro argument.
; CHECK-LABEL: swiftcoro_passthrough:
; OPT-NOT: mov{{.*}}x23
; OPT: bl {{_?}}swiftcoro_param
; OPT-NOT: mov{{.*}}x23
; OPT-NEXT: bl {{_?}}swiftcoro_param
; OPT: ret
define void @swiftcoro_passthrough(ptr swiftcoro %addr0) {
  call ptr @swiftcoro_param(ptr swiftcoro %addr0)
  call ptr @swiftcoro_param(ptr swiftcoro %addr0)
  ret void
}

; We can use a tail call if the callee swiftcoro is the same as the caller one.
; This should also work with fast-isel.
; CHECK-LABEL: swiftcoro_tail:
; OPTAARCH64: b {{_?}}swiftcoro_param
; OPTAARCH64-NOT: ret
; OPTARM64_32: b {{_?}}swiftcoro_param
define ptr @swiftcoro_tail(ptr swiftcoro %addr0) {
  call void asm sideeffect "", "~{x23}"()
  %res = musttail call ptr @swiftcoro_param(ptr swiftcoro %addr0)
  ret ptr %res
}

; We can not use a tail call if the callee swiftcoro is not the same as the
; caller one.
; CHECK-LABEL: swiftcoro_notail:
; CHECK: mov x23, x0
; CHECK: bl {{_?}}swiftcoro_param
; CHECK: ret
define ptr @swiftcoro_notail(ptr swiftcoro %addr0, ptr %addr1) nounwind {
  %res = tail call ptr @swiftcoro_param(ptr swiftcoro %addr1)
  ret ptr %res
}
