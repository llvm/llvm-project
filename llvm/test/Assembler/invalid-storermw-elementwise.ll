; RUN: split-file %s %t
; RUN: not llvm-as -disable-output %t/scalar.ll              2>&1 | FileCheck %t/scalar.ll
; RUN: not llvm-as -disable-output %t/odd-sized.ll           2>&1 | FileCheck %t/odd-sized.ll
; RUN: not llvm-as -disable-output %t/add-must-be-integer.ll 2>&1 | FileCheck %t/add-must-be-integer.ll
; RUN: not llvm-as -disable-output %t/fadd-must-be-fp.ll     2>&1 | FileCheck %t/fadd-must-be-fp.ll

;--- scalar.ll
; CHECK: storermw elementwise operand must be a fixed vector type
define void @bad_scalar(ptr %p, i32 %v) {
  storermw elementwise add ptr %p, i32 %v monotonic
  ret void
}

;--- odd-sized.ll
; CHECK: storermw operand must be power-of-two byte-sized integer
define void @bad_odd_sized_vector(ptr %p, <5 x i32> %v) {
  storermw elementwise add ptr %p, <5 x i32> %v monotonic, align 4
  ret void
}

;--- add-must-be-integer.ll
; CHECK: storermw add operand must be an integer
define void @bad_add(ptr %p, <4 x float> %v) {
  storermw elementwise add ptr %p, <4 x float> %v monotonic
  ret void
}

;--- fadd-must-be-fp.ll
; CHECK: storermw fadd operand must be a floating point type
define void @bad_fadd(ptr %p, <4 x i32> %v) {
  storermw elementwise fadd ptr %p, <4 x i32> %v monotonic
  ret void
}
