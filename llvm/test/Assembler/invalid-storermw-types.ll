; RUN: split-file %s %t
; RUN: not llvm-as -disable-output %t/add-fp-scalar.ll         2>&1 | FileCheck %t/add-fp-scalar.ll
; RUN: not llvm-as -disable-output %t/fadd-int-scalar.ll       2>&1 | FileCheck %t/fadd-int-scalar.ll
; RUN: not llvm-as -disable-output %t/i4-too-narrow.ll         2>&1 | FileCheck %t/i4-too-narrow.ll
; RUN: not llvm-as -disable-output %t/i9-not-pow2.ll           2>&1 | FileCheck %t/i9-not-pow2.ll
; RUN: not llvm-as -disable-output %t/scalable-vector.ll       2>&1 | FileCheck %t/scalable-vector.ll
; RUN: not llvm-as -disable-output %t/non-pointer-operand.ll   2>&1 | FileCheck %t/non-pointer-operand.ll

;--- add-fp-scalar.ll
; Non-elementwise integer op with a floating-point scalar operand.
; CHECK: storermw add operand must be an integer
define void @bad_add_fp(ptr %p, float %v) {
  storermw add ptr %p, float %v monotonic, align 4
  ret void
}

;--- fadd-int-scalar.ll
; Non-elementwise floating-point op with an integer scalar operand.
; CHECK: storermw fadd operand must be a floating point type
define void @bad_fadd_int(ptr %p, i32 %v) {
  storermw fadd ptr %p, i32 %v monotonic, align 4
  ret void
}

;--- i4-too-narrow.ll
; Value type must have bit width >= 8. Verifier catches this (the parser's
; check uses getTypeStoreSizeInBits which rounds i4 up to 8).
; CHECK: atomic memory access' size must be byte-sized
define void @bad_i4(ptr %p, i4 %v) {
  storermw add ptr %p, i4 %v monotonic, align 1
  ret void
}

;--- i9-not-pow2.ll
; Value type bit width must be a power of two. i9 has size >= 8 (so the
; "byte-sized" check passes) but is not a power of two, exercising the
; distinct power-of-two Verifier check.
; CHECK: atomic memory access' operand must have a power-of-two size
define void @bad_i9(ptr %p, i9 %v) {
  storermw add ptr %p, i9 %v monotonic, align 2
  ret void
}

;--- scalable-vector.ll
; Scalable vector value operand is not allowed.
; CHECK: storermw operand may not be scalable
define void @bad_scalable(ptr %p, <vscale x 4 x i32> %v) {
  storermw elementwise add ptr %p, <vscale x 4 x i32> %v monotonic, align 16
  ret void
}

;--- non-pointer-operand.ll
; The first operand must be a pointer, not (for example) an integer.
; CHECK: storermw operand must be a pointer
define void @bad_non_pointer(i32 %notptr, i32 %v) {
  storermw add i32 %notptr, i32 %v monotonic, align 4
  ret void
}
