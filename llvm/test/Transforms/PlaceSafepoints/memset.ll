; RUN: opt < %s -S -place-safepoints | FileCheck %s

define void @test(i32, i8 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test
; CHECK-NEXT: llvm.memset
; CHECK: do_safepoint
; CHECK: @foo
  call void @llvm.memset.p1i8.i64(i8 addrspace(1)* %ptr, i8 0, i64 24, i32 8, i1 false)
  call void @foo()
  ret void
}

declare void @foo()
declare void @llvm.memset.p1i8.i64(i8 addrspace(1)*, i8, i64, i32, i1)

declare void @do_safepoint()
define void @gc.safepoint_poll() {
  call void @do_safepoint()
  ret void
}
