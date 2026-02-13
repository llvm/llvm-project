; RUN: llc -mtriple=aarch64 -mattr=+pcdphint < %s | FileCheck %s

declare void @llvm.aarch64.stshh(i64)

define void @test_stshh_atomic_store(ptr %p, i32 %v) {
; CHECK-LABEL: test_stshh_atomic_store
; CHECK: stshh
; CHECK: str
  call void @llvm.aarch64.stshh(i64 0)
  store atomic i32 %v, ptr %p monotonic, align 4
  ret void
}
