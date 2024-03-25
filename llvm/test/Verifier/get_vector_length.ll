; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare i32 @llvm.experimental.get.vector.length.i32(i32, i32, i1)

define i32 @vector_length_negative_vf(i32 zeroext %tc) {
  ; CHECK: get_vector_length: VF must be positive
  ; CHECK-NEXT: %a = call i32 @llvm.experimental.get.vector.length.i32(i32 %tc, i32 -1, i1 true)
  %a = call i32 @llvm.experimental.get.vector.length.i32(i32 %tc, i32 -1, i1 true)
  ret i32 %a
}

define i32 @vector_length_zero_vf(i32 zeroext %tc) {
  ; CHECK: get_vector_length: VF must be positive
  ; CHECK-NEXT: %a = call i32 @llvm.experimental.get.vector.length.i32(i32 %tc, i32 0, i1 true)
  %a = call i32 @llvm.experimental.get.vector.length.i32(i32 %tc, i32 0, i1 true)
  ret i32 %a
}
