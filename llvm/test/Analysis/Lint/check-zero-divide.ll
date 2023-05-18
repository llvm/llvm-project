; RUN: opt -passes=lint -disable-output %s 2>&1 | FileCheck %s

define <2 x i32> @use_vector_sdiv(<2 x i32> %a) nounwind {
  %b = sdiv <2 x i32> %a, <i32 5, i32 8>
  ret <2 x i32> %b
}

define <2 x i32> @use_vector_srem(<2 x i32> %a) nounwind {
  %b = srem <2 x i32> %a, <i32 5, i32 8>
  ret <2 x i32> %b
}

define <2 x i32> @use_vector_udiv(<2 x i32> %a) nounwind {
  %b = udiv <2 x i32> %a, <i32 5, i32 8>
  ret <2 x i32> %b
}

define <2 x i32> @use_vector_urem(<2 x i32> %a) nounwind {
  %b = urem <2 x i32> %a, <i32 5, i32 8>
  ret <2 x i32> %b
}

define i32 @use_sdiv_by_zero(i32 %a) nounwind {
; CHECK: Undefined behavior: Division by zero
; CHECK-NEXT: %b = sdiv i32 %a, 0
  %b = sdiv i32 %a, 0
  ret i32 %b
}

define i32 @use_sdiv_by_zeroinitializer(i32 %a) nounwind {
; CHECK: Undefined behavior: Division by zero
; CHECK-NEXT: %b = sdiv i32 %a, 0
  %b = sdiv i32 %a, zeroinitializer
   ret i32 %b
}

define <2 x i32> @use_vector_sdiv_by_zero_x(<2 x i32> %a) nounwind {
; CHECK: Undefined behavior: Division by zero
; CHECK-NEXT: %b = sdiv <2 x i32> %a, <i32 0, i32 5>
  %b = sdiv <2 x i32> %a, <i32 0, i32 5>
  ret <2 x i32> %b
}

define <2 x i32> @use_vector_sdiv_by_zero_y(<2 x i32> %a) nounwind {
; CHECK: Undefined behavior: Division by zero
; CHECK-NEXT:  %b = sdiv <2 x i32> %a, <i32 4, i32 0>
  %b = sdiv <2 x i32> %a, <i32 4, i32 0>
  ret <2 x i32> %b
}

define <2 x i32> @use_vector_sdiv_by_zero_xy(<2 x i32> %a) nounwind {
; CHECK: Undefined behavior: Division by zero
; CHECK-NEXT: %b = sdiv <2 x i32> %a, zeroinitializer
  %b = sdiv <2 x i32> %a, <i32 0, i32 0>
  ret <2 x i32> %b
}

define <2 x i32> @use_vector_sdiv_by_undef_x(<2 x i32> %a) nounwind {
; CHECK: Undefined behavior: Division by zero
; CHECK-NEXT: %b = sdiv <2 x i32> %a, <i32 undef, i32 5>
  %b = sdiv <2 x i32> %a, <i32 undef, i32 5>
  ret <2 x i32> %b
}

define <2 x i32> @use_vector_sdiv_by_undef_y(<2 x i32> %a) nounwind {
; CHECK: Undefined behavior: Division by zero
; CHECK-NEXT: %b = sdiv <2 x i32> %a, <i32 5, i32 undef>
  %b = sdiv <2 x i32> %a, <i32 5, i32 undef>
  ret <2 x i32> %b
}

define <2 x i32> @use_vector_sdiv_by_undef_xy(<2 x i32> %a) nounwind {
; CHECK: Undefined behavior: Division by zero
; CHECK-NEXT: %b = sdiv <2 x i32> %a, undef
  %b = sdiv <2 x i32> %a, <i32 undef, i32 undef>
  ret <2 x i32> %b
}

