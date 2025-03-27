; RUN: opt -passes=mergefunc -S < %s | FileCheck %s

; CHECK-LABEL: @int_ptr_arg_different
; CHECK-NEXT: call void asm

; CHECK-DAG: @int_ptr_null
; CHECK-DAG: tail call void @float_ptr_null()

; CHECK-DAG: @int_ptr_arg_same
; CHECK-DAG: tail call void @float_ptr_arg_same(ptr %0)

; Used to satisfy minimum size limit
declare void @stuff()

; Can be merged
define void @float_ptr_null() {
  call void asm "nop", "r"(ptr null)
  call void @stuff()
  ret void
}

define void @int_ptr_null() {
  call void asm "nop", "r"(ptr null)
  call void @stuff()
  ret void
}

; Can be merged (uses same argument differing by pointer type)
define void @float_ptr_arg_same(ptr) {
  call void asm "nop", "r"(ptr %0)
  call void @stuff()
  ret void
}

define void @int_ptr_arg_same(ptr) {
  call void asm "nop", "r"(ptr %0)
  call void @stuff()
  ret void
}

; Can not be merged (uses different arguments)
define void @float_ptr_arg_different(ptr, ptr) {
  call void asm "nop", "r"(ptr %0)
  call void @stuff()
  ret void
}

define void @int_ptr_arg_different(ptr, ptr) {
  call void asm "nop", "r"(ptr %1)
  call void @stuff()
  ret void
}
