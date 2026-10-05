; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 -fast-isel < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 -global-isel -global-isel-abort=1 < %s | FileCheck %s

; The sanitizer region intrinsics are consumed by sanitizer passes, but they can
; still reach codegen when no such pass runs. All three instruction selectors
; have to drop them instead of failing to select.

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()

define void @test_sanitizer_intrinsics() {
; CHECK-LABEL: test_sanitizer_intrinsics:
; CHECK-NOT: call
; CHECK: retq
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}
