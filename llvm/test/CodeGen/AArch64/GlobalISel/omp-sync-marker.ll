; RUN: llc -mtriple=aarch64-unknown-linux-gnu -global-isel < %s | FileCheck %s

; llvm.omp.sync.marker doesn't generate any code.

define void @f(ptr %p) {
; CHECK-LABEL: f:
; CHECK-NOT: bl
; CHECK: ret
  call void @llvm.omp.sync.marker.p0(ptr %p)
  ret void
}
