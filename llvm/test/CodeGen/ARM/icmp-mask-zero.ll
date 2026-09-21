; RUN: llc -mtriple=thumbv8m.main-none-eabi -mcpu=cortex-m33 < %s | FileCheck %s

; Low contiguous masks can be shifted out before testing for zero. This avoids
; clearing the masked bits with BIC.

define i1 @is_zero_or_int_min_i32(i32 %x) {
; CHECK-LABEL: is_zero_or_int_min_i32:
; CHECK-NOT:    bic
; CHECK:        lsls r0, r0, #1
; CHECK-NEXT:   clz r0, r0
; CHECK-NEXT:   lsrs r0, r0, #5
; CHECK-NEXT:   bx lr
  %masked = and i32 %x, 2147483647
  %cmp = icmp eq i32 %masked, 0
  ret i1 %cmp
}

define i1 @is_zero_low_30_bits_i32(i32 %x) {
; CHECK-LABEL: is_zero_low_30_bits_i32:
; CHECK-NOT:    bic
; CHECK:        lsls r0, r0, #2
; CHECK-NEXT:   clz r0, r0
; CHECK-NEXT:   lsrs r0, r0, #5
; CHECK-NEXT:   bx lr
  %masked = and i32 %x, 1073741823
  %cmp = icmp eq i32 %masked, 0
  ret i1 %cmp
}

define i1 @is_zero_low_8_bits_i32(i32 %x) {
; CHECK-LABEL: is_zero_low_8_bits_i32:
; CHECK:        uxtb r0, r0
; CHECK-NEXT:   clz r0, r0
; CHECK-NEXT:   lsrs r0, r0, #5
; CHECK-NEXT:   bx lr
  %masked = and i32 %x, 255
  %cmp = icmp eq i32 %masked, 0
  ret i1 %cmp
}

define i1 @is_zero_low_16_bits_i32(i32 %x) {
; CHECK-LABEL: is_zero_low_16_bits_i32:
; CHECK:        uxth r0, r0
; CHECK-NEXT:   clz r0, r0
; CHECK-NEXT:   lsrs r0, r0, #5
; CHECK-NEXT:   bx lr
  %masked = and i32 %x, 65535
  %cmp = icmp eq i32 %masked, 0
  ret i1 %cmp
}

define i1 @is_zero_or_int_min_i64(i64 %x) {
; CHECK-LABEL: is_zero_or_int_min_i64:
; CHECK-NOT:    bic
; CHECK:        lsls r1, r1, #1
; CHECK-NEXT:   orrs r0, r1
; CHECK-NEXT:   clz r0, r0
; CHECK-NEXT:   lsrs r0, r0, #5
; CHECK-NEXT:   bx lr
  %masked = and i64 %x, 9223372036854775807
  %cmp = icmp eq i64 %masked, 0
  ret i1 %cmp
}

declare void @func_1_i32(i32)
declare void @func_2_i32(i32)

define void @branch_on_zero_or_int_min_i32(i32 %x) {
; CHECK-LABEL: branch_on_zero_or_int_min_i32:
; CHECK:        lsls r1, r0, #1
; CHECK-NEXT:   it ne
; CHECK-NEXT:   bne func_2_i32
; CHECK:        b func_1_i32
  %masked = and i32 %x, 2147483647
  %cmp = icmp eq i32 %masked, 0
  br i1 %cmp, label %yes, label %no

yes:
  tail call void @func_1_i32(i32 %x)
  ret void

no:
  tail call void @func_2_i32(i32 %x)
  ret void
}

declare void @func_1_i64(i64)
declare void @func_2_i64(i64)

define void @branch_on_zero_or_int_min_i64(i64 %x) {
; CHECK-LABEL: branch_on_zero_or_int_min_i64:
; CHECK:        lsls r2, r1, #1
; CHECK-NEXT:   orrs r2, r0
; CHECK-NEXT:   it ne
; CHECK-NEXT:   bne func_2_i64
; CHECK:        b func_1_i64
  %masked = and i64 %x, 9223372036854775807
  %cmp = icmp eq i64 %masked, 0
  br i1 %cmp, label %yes, label %no

yes:
  tail call void @func_1_i64(i64 %x)
  ret void

no:
  tail call void @func_2_i64(i64 %x)
  ret void
}
