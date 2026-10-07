; RUN: llc -mtriple=thumbv8m.main-none-eabi -mcpu=cortex-m33 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=thumbv7m-none-eabi -mcpu=cortex-m3 -verify-machineinstrs < %s | FileCheck %s

; Shift out the sign bit instead of clearing it with BIC before the zero test.
define i1 @is_zero_or_int_min_i32(i32 %x) {
; CHECK-LABEL: is_zero_or_int_min_i32:
; CHECK:       lsls r0, r0, #1
; CHECK-NEXT:  clz r0, r0
; CHECK-NEXT:  lsrs r0, r0, #5
; CHECK-NEXT:  bx lr
  %masked = and i32 %x, 2147483647
  %cmp = icmp eq i32 %masked, 0
  ret i1 %cmp
}

; The same transform works for a different low mask inside an OR.
define i1 @is_zero_low_30_bits_or_i32(i32 %x, i32 %y) {
; CHECK-LABEL: is_zero_low_30_bits_or_i32:
; CHECK:       orr.w r0, r1, r0, lsl #2
; CHECK-NEXT:  clz r0, r0
; CHECK-NEXT:  lsrs r0, r0, #5
; CHECK-NEXT:  bx lr
  %masked = and i32 %x, 1073741823
  %combined = or i32 %y, %masked
  %cmp = icmp eq i32 %combined, 0
  ret i1 %cmp
}

; i64 legalization ORs the halves together. Fold the shift into that OR.
define i1 @is_zero_or_int_min_i64(i64 %x) {
; CHECK-LABEL: is_zero_or_int_min_i64:
; CHECK:       orr.w r0, r0, r1, lsl #1
; CHECK-NEXT:  clz r0, r0
; CHECK-NEXT:  lsrs r0, r0, #5
; CHECK-NEXT:  bx lr
  %masked = and i64 %x, 9223372036854775807
  %cmp = icmp eq i64 %masked, 0
  ret i1 %cmp
}

declare void @func_1_i64(i64)
declare void @func_2_i64(i64)

; Also use the shifted OR when branching on inequality against zero.
define void @branch_on_nonzero_low_63_bits_i64(i64 %x) {
; CHECK-LABEL: branch_on_nonzero_low_63_bits_i64:
; CHECK:       orrs.w r2, r0, r1, lsl #1
; CHECK-NEXT:  it eq
; CHECK-NEXT:  beq func_2_i64
; CHECK:       b func_1_i64
  %masked = and i64 %x, 9223372036854775807
  %cmp = icmp ne i64 %masked, 0
  br i1 %cmp, label %yes, label %no

yes:
  tail call void @func_1_i64(i64 %x)
  ret void

no:
  tail call void @func_2_i64(i64 %x)
  ret void
}

; The transform preserves zero/nonzero, not comparisons against other values.
define i1 @masked_or_equals_nonzero(i32 %x, i32 %y) {
; CHECK-LABEL: masked_or_equals_nonzero:
; CHECK:       bic r0, r0, #-2147483648
; CHECK-NEXT:  orrs r0, r1
; CHECK-NEXT:  subs r0, #1
  %masked = and i32 %x, 2147483647
  %combined = or i32 %masked, %y
  %cmp = icmp eq i32 %combined, 1
  ret i1 %cmp
}

; A shift can change the sign, so keep the mask for signed comparisons.
define i1 @masked_or_signed_greater_than_zero(i32 %x, i32 %y) {
; CHECK-LABEL: masked_or_signed_greater_than_zero:
; CHECK:       bic r0, r0, #-2147483648
; CHECK-NEXT:  orrs r0, r1
; CHECK-NEXT:  cmp r0, #0
  %masked = and i32 %x, 2147483647
  %combined = or i32 %masked, %y
  %cmp = icmp sgt i32 %combined, 0
  ret i1 %cmp
}

; Shifting cannot remove a hole in the mask.
define i1 @is_zero_noncontiguous_masked_or(i32 %x, i32 %y) {
; CHECK-LABEL: is_zero_noncontiguous_masked_or:
; CHECK:       movw r2, #65533
; CHECK-NEXT:  movt r2, #32767
; CHECK-NEXT:  ands r0, r2
; CHECK-NEXT:  orrs r0, r1
  %masked = and i32 %x, 2147483645
  %combined = or i32 %masked, %y
  %cmp = icmp eq i32 %combined, 0
  ret i1 %cmp
}

; Reuse a live masked value rather than adding a separate shift for its test.
define i1 @is_zero_mask_multiple_uses(i32 %x, ptr %p) {
; CHECK-LABEL: is_zero_mask_multiple_uses:
; CHECK:       bic r2, r0, #-2147483648
; CHECK-NEXT:  clz r0, r2
; CHECK:       str r2, [r1]
  %masked = and i32 %x, 2147483647
  store i32 %masked, ptr %p
  %cmp = icmp eq i32 %masked, 0
  ret i1 %cmp
}
