; RUN: llc < %s -mtriple=i686-none-linux -verify-machineinstrs | FileCheck %s

define ptr @test_atomic_ptr_load(ptr %a0) {
; CHECK: test_atomic_ptr_load
; CHECK: movl
; CHECK: movl
; CHECK: ret
entry:
  %0 = load atomic ptr, ptr %a0 seq_cst, align 4
  ret ptr %0
}

define void @test_atomic_ptr_store(ptr %a0, ptr %a1) {
; CHECK: test_atomic_ptr_store
; CHECK: movl
; CHECK: movl
; CHECK: xchgl
; CHECK: ret
entry:
store atomic ptr %a0, ptr %a1 seq_cst, align 4
  ret void
}
