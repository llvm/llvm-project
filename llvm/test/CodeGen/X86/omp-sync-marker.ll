; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 < %s | FileCheck %s

; llvm.omp.sync.marker doesn't generate any code.

define void @f(ptr %p) {
; CHECK-LABEL: f:
; CHECK-NOT: call
; CHECK: retq
  call void @llvm.omp.sync.marker.p0(ptr %p)
  ret void
}
