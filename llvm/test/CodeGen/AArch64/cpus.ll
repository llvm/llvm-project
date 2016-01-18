; This tests that llc accepts all valid AArch64 CPUs


; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=generic 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a53 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a57 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=cortex-a72 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-unknown-unknown -mcpu=invalidcpu 2>&1 | FileCheck %s --check-prefix=INVALID

; CHECK-NOT: {{.*}}  is not a recognized processor for this target
; INVALID: {{.*}}  is not a recognized processor for this target

define i32 @f(i64 %z) {
	ret i32 0
}
