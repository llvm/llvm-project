; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s
; RUN: llvm-as %s -o - | verify-uselistorder

define <4 x i32> @elem_add(ptr %p, <4 x i32> %v) {
; CHECK-LABEL: @elem_add(
; CHECK: %old = atomicrmw elementwise add ptr %p, <4 x i32> %v monotonic, align 16
  %old = atomicrmw elementwise add ptr %p, <4 x i32> %v monotonic
  ret <4 x i32> %old
}

define <4 x float> @elem_fadd(ptr %p, <4 x float> %v) {
; CHECK-LABEL: @elem_fadd(
; CHECK: %old = atomicrmw elementwise fadd ptr %p, <4 x float> %v seq_cst, align 16
  %old = atomicrmw elementwise fadd ptr %p, <4 x float> %v seq_cst
  ret <4 x float> %old
}
