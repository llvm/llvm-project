; RUN: split-file %s %t
; RUN: not llvm-as -disable-output %t/load-non-atomic.ll 2>&1 | FileCheck %t/load-non-atomic.ll
; RUN: not llvm-as -disable-output %t/load-scalar.ll 2>&1 | FileCheck %t/load-scalar.ll
; RUN: not llvm-as -disable-output %t/load-scalable.ll 2>&1 | FileCheck %t/load-scalable.ll
; RUN: not llvm-as -disable-output %t/load-odd-sized.ll 2>&1 | FileCheck %t/load-odd-sized.ll
; RUN: not llvm-as -disable-output %t/load-non-byte.ll 2>&1 | FileCheck %t/load-non-byte.ll
; RUN: not llvm-as -disable-output %t/load-non-byte-element.ll 2>&1 | FileCheck %t/load-non-byte-element.ll

;--- load-non-atomic.ll
; CHECK: elementwise load must be atomic
define <2 x float> @bad_non_atomic(ptr %p) {
  %v = load elementwise <2 x float>, ptr %p, align 4
  ret <2 x float> %v
}

;--- load-scalar.ll
; CHECK: atomic elementwise load operand must have fixed vector type
define float @bad_scalar(ptr %p) {
  %v = load atomic elementwise float, ptr %p monotonic, align 4
  ret float %v
}

;--- load-scalable.ll
; CHECK: atomic elementwise load operand must have fixed vector type
define <vscale x 2 x i32> @bad_scalable(ptr %p) {
  %v = load atomic elementwise <vscale x 2 x i32>, ptr %p monotonic, align 4
  ret <vscale x 2 x i32> %v
}

;--- load-odd-sized.ll
; CHECK: atomic memory access' operand must have a power-of-two size
define <5 x i32> @bad_odd_sized_vector(ptr %p) {
  %v = load atomic elementwise <5 x i32>, ptr %p monotonic, align 4
  ret <5 x i32> %v
}

;--- load-non-byte.ll
; CHECK: atomic memory access' size must be byte-sized
define <4 x i1> @bad_non_byte(ptr %p) {
  %v = load atomic elementwise <4 x i1>, ptr %p monotonic, align 4
  ret <4 x i1> %v
}

;--- load-non-byte-element.ll
; CHECK: atomic memory access' size must be byte-sized
define <8 x i1> @bad_non_byte_element(ptr %p) {
  %v = load atomic elementwise <8 x i1>, ptr %p monotonic, align 1
  ret <8 x i1> %v
}
