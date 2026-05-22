; RUN: split-file %s %t
; RUN: not llvm-as -disable-output %t/scalar.ll              2>&1 | FileCheck %t/scalar.ll
; RUN: not llvm-as -disable-output %t/odd-sized.ll           2>&1 | FileCheck %t/odd-sized.ll
; RUN: not llvm-as -disable-output %t/add-must-be-integer.ll 2>&1 | FileCheck %t/add-must-be-integer.ll
; RUN: not llvm-as -disable-output %t/fadd-must-be-fp.ll     2>&1 | FileCheck %t/fadd-must-be-fp.ll

;--- scalar.ll
; CHECK: atomicrmw elementwise operand must be a fixed vector type
define i32 @bad_scalar(ptr %p, i32 %v) {
  %old = atomicrmw elementwise add ptr %p, i32 %v monotonic
  ret i32 %old
}

;--- odd-sized.ll
; CHECK: atomicrmw operand must have a power-of-two byte size
define <5 x i32> @bad_odd_sized_vector(ptr %p, <5 x i32> %v) {
  %old = atomicrmw elementwise add ptr %p, <5 x i32> %v monotonic, align 4
  ret <5 x i32> %old
}

;--- add-must-be-integer.ll
; CHECK: atomicrmw add operand must be an integer
define <4 x float> @bad_add(ptr %p, <4 x float> %v) {
  %old = atomicrmw elementwise add ptr %p, <4 x float> %v monotonic
  ret <4 x float> %old
}

;--- fadd-must-be-fp.ll
; CHECK: atomicrmw fadd operand must be a floating point type
define <4 x i32> @bad_fadd(ptr %p, <4 x i32> %v) {
  %old = atomicrmw elementwise fadd ptr %p, <4 x i32> %v monotonic
  ret <4 x i32> %old
}
