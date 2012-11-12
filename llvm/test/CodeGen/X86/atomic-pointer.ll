; RUN: llc < %s -mtriple=i686-none-linux | FileCheck %s

define i32* @test_atomic_ptr_load(i32** %a0) {
; CHECK: test_atomic_ptr_load
; CHECK: movl
; CHECK: movl
; CHECK: ret
0:
  %0 = load atomic i32** %a0 seq_cst, align 4
  ret i32* %0
}

define void @test_atomic_ptr_store(i32* %a0, i32** %a1) {
; CHECK: test_atomic_ptr_store
; CHECK: movl
; CHECK: movl
; CHECK: xchgl
; CHECK: ret
0:
  store atomic i32* %a0, i32** %a1 seq_cst, align 4
  ret void
}
