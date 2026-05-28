; RUN: llc -O0 < %s | FileCheck %s
; RUN: llc -O2 < %s | FileCheck %s
; RUN: llc -O0 -fast-isel < %s | FileCheck %s

; Verify that the backend successfully lowers KMSAN region intrinsics to nothing
; and does not crash when compiling a module where MSan did not run.

declare void @llvm.kmsan.instrumentation.begin()
declare void @llvm.kmsan.instrumentation.update.context()
declare void @llvm.kmsan.instrumentation.end()

define void @test_kmsan_region_intrinsics() {
; CHECK-LABEL: test_kmsan_region_intrinsics:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ret{{[ql]?}}
entry:
  call void @llvm.kmsan.instrumentation.begin()
  call void @llvm.kmsan.instrumentation.update.context()
  call void @llvm.kmsan.instrumentation.end()
  ret void
}