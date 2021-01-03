; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; Check that we convert
;   trunc(C * a) -> trunc(C) * trunc(a)
; if C is a constant.
; CHECK-LABEL: @trunc_of_mul
define i8 @trunc_of_mul(i32 %a) {
  %b = mul i32 %a, 100
  ; CHECK: %c
  ; CHECK-NEXT: --> (100 * (trunc i32 %a to i8))
  %c = trunc i32 %b to i8
  ret i8 %c
}

; Check that we convert
;   trunc(C + a) -> trunc(C) + trunc(a)
; if C is a constant.
; CHECK-LABEL: @trunc_of_add
define i8 @trunc_of_add(i32 %a) {
  %b = add i32 %a, 100
  ; CHECK: %c
  ; CHECK-NEXT: --> (100 + (trunc i32 %a to i8))
  %c = trunc i32 %b to i8
  ret i8 %c
}

; Check that we truncate to zero values assumed to have at least as many
; trailing zeros as the target type.
; CHECK-LABEL: @trunc_to_assumed_zeros
define i8 @trunc_to_assumed_zeros(i32* %p) {
  %a = load i32, i32* %p
  %and = and i32 %a, 255
  %cmp = icmp eq i32 %and, 0
  tail call void @llvm.assume(i1 %cmp)
  ; CHECK: %c
  ; CHECK-NEXT: --> 0
  %c = trunc i32 %a to i8
  ; CHECK: %d
  ; CHECK-NEXT: --> false
  %d = trunc i32 %a to i1
  ; CHECK: %e
  ; CHECK-NEXT: --> (trunc i32 %a to i16)
  %e = trunc i32 %a to i16
  ret i8 %c
}

declare void @llvm.assume(i1 noundef) nofree nosync nounwind willreturn
