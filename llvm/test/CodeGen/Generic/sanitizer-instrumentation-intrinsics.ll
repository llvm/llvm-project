; RUN: llc -O0 < %s | FileCheck %s
; RUN: llc -O2 < %s | FileCheck %s
; RUN: llc -O0 -fast-isel < %s | FileCheck %s

; Verify that the backend lowers the sanitizer region intrinsics to nothing and
; does not crash when compiling a module where no sanitizer pass ran.

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()

define void @test_sanitizer_intrinsics() {
; CHECK-LABEL: test_sanitizer_intrinsics:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ret{{[ql]?}}
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}
