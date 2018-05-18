; RUN: opt -mergefunc -S < %s | FileCheck %s

; CHECK-LABEL: @int_ptr_arg_different
; CHECK-NEXT: call void asm

; CHECK-LABEL: @int_ptr_arg_same
; CHECK-NEXT: %2 = bitcast i32* %0 to float*
; CHECK-NEXT: tail call void @float_ptr_arg_same(float* %2)

; CHECK-LABEL: @int_ptr_null
; CHECK-NEXT: tail call void @float_ptr_null()

; Used to satisfy minimum size limit
declare void @stuff()

; Can be merged
define void @float_ptr_null() {
  call void asm "nop", "r"(float* null)
  call void @stuff()
  ret void
}

define void @int_ptr_null() {
  call void asm "nop", "r"(i32* null)
  call void @stuff()
  ret void
}

; Can be merged (uses same argument differing by pointer type)
define void @float_ptr_arg_same(float*) {
  call void asm "nop", "r"(float* %0)
  call void @stuff()
  ret void
}

define void @int_ptr_arg_same(i32*) {
  call void asm "nop", "r"(i32* %0)
  call void @stuff()
  ret void
}

; Can not be merged (uses different arguments)
define void @float_ptr_arg_different(float*, float*) {
  call void asm "nop", "r"(float* %0)
  call void @stuff()
  ret void
}

define void @int_ptr_arg_different(i32*, i32*) {
  call void asm "nop", "r"(i32* %1)
  call void @stuff()
  ret void
}
