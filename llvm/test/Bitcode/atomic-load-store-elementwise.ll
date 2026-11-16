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
; CHECK: %v = load atomic volatile elementwise <4 x i32>, ptr %p acquire, align 4
  %v = load atomic volatile elementwise <4 x i32>, ptr %p acquire, align 4
  ret <4 x i32> %v
}

define void @store_elem_f32(ptr %p, <2 x float> %v) {
; CHECK-LABEL: @store_elem_f32(
; CHECK: store atomic elementwise <2 x float> %v, ptr %p syncscope("agent") monotonic, align 4
  store atomic elementwise <2 x float> %v, ptr %p syncscope("agent") monotonic, align 4
  ret void
}

define void @store_elem_i32_volatile(ptr %p, <4 x i32> %v) {
; CHECK-LABEL: @store_elem_i32_volatile(
; CHECK: store atomic volatile elementwise <4 x i32> %v, ptr %p monotonic, align 4
  store atomic volatile elementwise <4 x i32> %v, ptr %p monotonic, align 4
  ret void
}
