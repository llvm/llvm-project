; RUN: llc -mtriple=avr < %s | FileCheck %s

; CHECK-LABEL: ret_void_args_i8
define void @ret_void_args_i8(i8 %a) {
  ; CHECK: sts 4, r24
  store volatile i8 %a, ptr inttoptr (i64 4 to ptr)
  ret void
}

; CHECK-LABEL: ret_void_args_i8_i32
define void @ret_void_args_i8_i32(i8 %a, i32 %b) {
  ; CHECK:      sts     4, r24
  store volatile i8 %a, ptr inttoptr (i64 4 to ptr)

  ; CHECK-NEXT: sts     8, r23
  ; CHECK-NEXT: sts     7, r22
  ; CHECK-NEXT: sts     6, r21
  ; CHECK-NEXT: sts     5, r20
  store volatile i32 %b, ptr inttoptr (i64 5 to ptr)
  ret void
}

; CHECK-LABEL: ret_void_args_i8_i8_i8_i8
define void @ret_void_args_i8_i8_i8_i8(i8 %a, i8 %b, i8 %c, i8 %d) {
  ; CHECK:      sts     4, r24
  store volatile i8 %a, ptr inttoptr (i64 4 to ptr)
  ; CHECK-NEXT: sts     5, r22
  store volatile i8 %b, ptr inttoptr (i64 5 to ptr)
  ; CHECK-NEXT: sts     6, r20
  store volatile i8 %c, ptr inttoptr (i64 6 to ptr)
  ; CHECK-NEXT: sts     7, r18
  store volatile i8 %d, ptr inttoptr (i64 7 to ptr)
  ret void
}

; CHECK-LABEL: ret_void_args_i32_16_i8
define void @ret_void_args_i32_16_i8(i32 %a, i16 %b, i8 %c) {
  ; CHECK:      sts     7, r25
  ; CHECK-NEXT: sts     6, r24
  ; CHECK-NEXT: sts     5, r23
  ; CHECK-NEXT: sts     4, r22
  store volatile i32 %a, ptr inttoptr (i64 4 to ptr)

  ; CHECK-NEXT: sts     5, r21
  ; CHECK-NEXT: sts     4, r20
  store volatile i16 %b, ptr inttoptr (i64 4 to ptr)

  ; CHECK-NEXT: sts     4, r18
  store volatile i8 %c, ptr inttoptr (i64 4 to ptr)
  ret void
}

; CHECK-LABEL: ret_void_args_i64
define void @ret_void_args_i64(i64 %a) {
  ; CHECK:      sts     11, r25
  ; CHECK-NEXT: sts     10, r24
  ; CHECK-NEXT: sts     9, r23
  ; CHECK-NEXT: sts     8, r22
  ; CHECK-NEXT: sts     7, r21
  ; CHECK-NEXT: sts     6, r20
  ; CHECK-NEXT: sts     5, r19
  ; CHECK-NEXT: sts     4, r18
  store volatile i64 %a, ptr inttoptr (i64 4 to ptr)
  ret void
}

; CHECK-LABEL: ret_void_args_i64_i64
define void @ret_void_args_i64_i64(i64 %a, i64 %b) {
  ; CHECK-DAG:  sts     11, r25
  ; CHECK-DAG: sts     10, r24
  ; CHECK-DAG: sts     9, r23
  ; CHECK-DAG: sts     8, r22
  ; CHECK-DAG: sts     7, r21
  ; CHECK-DAG: sts     6, r20
  ; CHECK-DAG: sts     5, r19
  ; CHECK-DAG: sts     4, r18
  store volatile i64 %a, ptr inttoptr (i64 4 to ptr)

  ; CHECK-DAG: sts     11, r17
  ; CHECK-DAG: sts     10, r16
  ; CHECK-DAG: sts     9, r15
  ; CHECK-DAG: sts     8, r14
  ; CHECK-DAG: sts     7, r13
  ; CHECK-DAG: sts     6, r12
  ; CHECK-DAG: sts     5, r11
  ; CHECK-DAG: sts     4, r10
  store volatile i64 %b, ptr inttoptr (i64 4 to ptr)
  ret void
}

; This is exactly enough to hit the limit of what can be passed
; completely in registers.
; CHECK-LABEL: ret_void_args_i64_i64_i16
define void @ret_void_args_i64_i64_i16(i64 %a, i64 %b, i16 %c) {
  ; CHECK:      sts     5, r9
  ; CHECK-NEXT: sts     4, r8
  store volatile i16 %c, ptr inttoptr (i64 4 to ptr)
  ret void
}

; NOTE: Both %a (i8) and %b (i8) cost two registers.
define i8 @foo0(i8 %a, i8 %b) {
; CHECK-LABEL: foo0:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    sub r24, r22
; CHECK-NEXT:    ret
  %c = sub i8 %a, %b
  ret i8 %c
}

; NOTE: Both %a (i16) and %b (i16) cost two registers.
define i16 @foo1(i16 %a, i16 %b) {
; CHECK-LABEL: foo1:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    sub r24, r22
; CHECK-NEXT:    sbc r25, r23
; CHECK-NEXT:    ret
  %c = sub i16 %a, %b
  ret i16 %c
}

; NOTE: Both %a (i32) and %b (i32) cost four registers.
define i32 @foo2(i32 %a, i32 %b) {
; CHECK-LABEL: foo2:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    sub r22, r18
; CHECK-NEXT:    sbc r23, r19
; CHECK-NEXT:    sbc r24, r20
; CHECK-NEXT:    sbc r25, r21
; CHECK-NEXT:    ret
  %c = sub i32 %a, %b
  ret i32 %c
}

; NOTE: Each argument costs four registers, and total 16 registers are used.
define i32 @foo3(i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK-LABEL: foo3:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    sub r22, r10
; CHECK-NEXT:    sbc r23, r11
; CHECK-NEXT:    sbc r24, r12
; CHECK-NEXT:    sbc r25, r13
; CHECK-NEXT:    ret
  %e = sub nsw i32 %a, %d
  ret i32 %e
}

; NOTE: Each argument (except %e) cost four registers, and total 16 registers
; NOTE: are used. Though there are still 2 registers are vacant, the %e has
; NOTE: to be dropped to the stack.
define i32 @foo4(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
; CHECK-LABEL: foo4:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    push r28
; CHECK-NEXT:    push r29
; CHECK-NEXT:    in r28, 61
; CHECK-NEXT:    in r29, 62
; CHECK-NEXT:    ldd r18, Y+5
; CHECK-NEXT:    ldd r19, Y+6
; CHECK-NEXT:    ldd r20, Y+7
; CHECK-NEXT:    ldd r21, Y+8
; CHECK-NEXT:    sub r22, r18
; CHECK-NEXT:    sbc r23, r19
; CHECK-NEXT:    sbc r24, r20
; CHECK-NEXT:    sbc r25, r21
; CHECK-NEXT:    pop r29
; CHECK-NEXT:    pop r28
; CHECK-NEXT:    ret
  %f = sub nsw i32 %a, %e
  ret i32 %f
}
