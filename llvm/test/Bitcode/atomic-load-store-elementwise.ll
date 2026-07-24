; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s
; RUN: llvm-as %s -o - | verify-uselistorder

define <2 x float> @load_elem_f32(ptr %p) {
; CHECK-LABEL: @load_elem_f32(
; CHECK: %v = load atomic elementwise <2 x float>, ptr %p syncscope("agent") monotonic, align 4
  %v = load atomic elementwise <2 x float>, ptr %p syncscope("agent") monotonic, align 4
  ret <2 x float> %v
}

define <4 x i32> @load_elem_i32_volatile(ptr %p) {
; CHECK-LABEL: @load_elem_i32_volatile(
; CHECK: %v = load atomic volatile elementwise <4 x i32>, ptr %p seq_cst, align 4
  %v = load atomic volatile elementwise <4 x i32>, ptr %p seq_cst, align 4
  ret <4 x i32> %v
}
