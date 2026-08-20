; RUN: split-file %s %t
; RUN: not llvm-as -disable-output %t/load-non-atomic.ll 2>&1 | FileCheck %t/load-non-atomic.ll
; RUN: not llvm-as -disable-output %t/load-scalar.ll 2>&1 | FileCheck %t/load-scalar.ll
; RUN: not llvm-as -disable-output %t/load-scalable.ll 2>&1 | FileCheck %t/load-scalable.ll
; RUN: not llvm-as -disable-output %t/load-odd-sized.ll 2>&1 | FileCheck %t/load-odd-sized.ll
; RUN: not llvm-as -disable-output %t/load-non-byte.ll 2>&1 | FileCheck %t/load-non-byte.ll
; RUN: not llvm-as -disable-output %t/load-non-byte-element.ll 2>&1 | FileCheck %t/load-non-byte-element.ll
; RUN: not llvm-as -disable-output %t/load-seq-cst.ll 2>&1 | FileCheck %t/load-seq-cst.ll
; RUN: not llvm-as -disable-output %t/store-non-atomic.ll 2>&1 | FileCheck %t/store-non-atomic.ll
; RUN: not llvm-as -disable-output %t/store-scalar.ll 2>&1 | FileCheck %t/store-scalar.ll
; RUN: not llvm-as -disable-output %t/store-scalable.ll 2>&1 | FileCheck %t/store-scalable.ll
; RUN: not llvm-as -disable-output %t/store-odd-sized.ll 2>&1 | FileCheck %t/store-odd-sized.ll
; RUN: not llvm-as -disable-output %t/store-non-byte.ll 2>&1 | FileCheck %t/store-non-byte.ll
; RUN: not llvm-as -disable-output %t/store-non-byte-element.ll 2>&1 | FileCheck %t/store-non-byte-element.ll
; RUN: not llvm-as -disable-output %t/store-seq-cst.ll 2>&1 | FileCheck %t/store-seq-cst.ll

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

;--- load-seq-cst.ll
; CHECK: atomic elementwise load cannot be sequentially consistent
define <4 x i32> @bad_seq_cst(ptr %p) {
  %v = load atomic elementwise <4 x i32>, ptr %p seq_cst, align 4
  ret <4 x i32> %v
}

;--- store-non-atomic.ll
; CHECK: elementwise store must be atomic
define void @bad_non_atomic_store(ptr %p, <2 x float> %v) {
  store elementwise <2 x float> %v, ptr %p, align 4
  ret void
}

;--- store-scalar.ll
; CHECK: atomic elementwise store operand must have fixed vector type
define void @bad_scalar_store(ptr %p, float %v) {
  store atomic elementwise float %v, ptr %p monotonic, align 4
  ret void
}

;--- store-scalable.ll
; CHECK: atomic elementwise store operand must have fixed vector type
define void @bad_scalable_store(ptr %p, <vscale x 2 x i32> %v) {
  store atomic elementwise <vscale x 2 x i32> %v, ptr %p monotonic, align 4
  ret void
}

;--- store-odd-sized.ll
; CHECK: atomic memory access' operand must have a power-of-two size
define void @bad_odd_sized_vector_store(ptr %p, <5 x i32> %v) {
  store atomic elementwise <5 x i32> %v, ptr %p monotonic, align 4
  ret void
}

;--- store-non-byte.ll
; CHECK: atomic memory access' size must be byte-sized
define void @bad_non_byte_store(ptr %p, <4 x i1> %v) {
  store atomic elementwise <4 x i1> %v, ptr %p monotonic, align 4
  ret void
}

;--- store-non-byte-element.ll
; CHECK: atomic memory access' size must be byte-sized
define void @bad_non_byte_element_store(ptr %p, <8 x i1> %v) {
  store atomic elementwise <8 x i1> %v, ptr %p monotonic, align 1
  ret void
}

;--- store-seq-cst.ll
; CHECK: atomic elementwise store cannot be sequentially consistent
define void @bad_store_seq_cst(ptr %p, <4 x i32> %v) {
  store atomic elementwise <4 x i32> %v, ptr %p seq_cst, align 4
  ret void
}
