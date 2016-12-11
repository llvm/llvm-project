; RUN: llc < %s -march=avr | FileCheck %s

; CHECK-LABEL: ret_void_args_i8
define void @ret_void_args_i8(i8 %a) {
  ; CHECK: sts 4, r24
  store volatile i8 %a, i8* inttoptr (i64 4 to i8*)
  ret void
}

; CHECK-LABEL: ret_void_args_i8_i32
define void @ret_void_args_i8_i32(i8 %a, i32 %b) {
  ; CHECK:      sts     4, r24
  store volatile i8 %a, i8* inttoptr (i64 4 to i8*)

  ; CHECK-NEXT: sts     8, r23
  ; CHECK-NEXT: sts     7, r22
  ; CHECK-NEXT: sts     6, r21
  ; CHECK-NEXT: sts     5, r20
  store volatile i32 %b, i32* inttoptr (i64 5 to i32*)
  ret void
}

; CHECK-LABEL: ret_void_args_i8_i8_i8_i8
define void @ret_void_args_i8_i8_i8_i8(i8 %a, i8 %b, i8 %c, i8 %d) {
  ; CHECK:      sts     4, r24
  store volatile i8 %a, i8* inttoptr (i64 4 to i8*)
  ; CHECK-NEXT: sts     5, r22
  store volatile i8 %b, i8* inttoptr (i64 5 to i8*)
  ; CHECK-NEXT: sts     6, r20
  store volatile i8 %c, i8* inttoptr (i64 6 to i8*)
  ; CHECK-NEXT: sts     7, r18
  store volatile i8 %d, i8* inttoptr (i64 7 to i8*)
  ret void
}

; CHECK-LABEL: ret_void_args_i32_16_i8
define void @ret_void_args_i32_16_i8(i32 %a, i16 %b, i8 %c) {
  ; CHECK:      sts     7, r25
  ; CHECK-NEXT: sts     6, r24
  ; CHECK-NEXT: sts     5, r23
  ; CHECK-NEXT: sts     4, r22
  store volatile i32 %a, i32* inttoptr (i64 4 to i32*)

  ; CHECK-NEXT: sts     5, r21
  ; CHECK-NEXT: sts     4, r20
  store volatile i16 %b, i16* inttoptr (i64 4 to i16*)

  ; CHECK-NEXT: sts     4, r18
  store volatile i8 %c, i8* inttoptr (i64 4 to i8*)
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
  store volatile i64 %a, i64* inttoptr (i64 4 to i64*)
  ret void
}

; CHECK-LABEL: ret_void_args_i64_i64
define void @ret_void_args_i64_i64(i64 %a, i64 %b) {
  ; CHECK:      sts     11, r25
  ; CHECK-NEXT: sts     10, r24
  ; CHECK-NEXT: sts     9, r23
  ; CHECK-NEXT: sts     8, r22
  ; CHECK-NEXT: sts     7, r21
  ; CHECK-NEXT: sts     6, r20
  ; CHECK-NEXT: sts     5, r19
  ; CHECK-NEXT: sts     4, r18
  store volatile i64 %a, i64* inttoptr (i64 4 to i64*)

  ; CHECK-NEXT: sts     11, r17
  ; CHECK-NEXT: sts     10, r16
  ; CHECK-NEXT: sts     9, r15
  ; CHECK-NEXT: sts     8, r14
  ; CHECK-NEXT: sts     7, r13
  ; CHECK-NEXT: sts     6, r12
  ; CHECK-NEXT: sts     5, r11
  ; CHECK-NEXT: sts     4, r10
  store volatile i64 %b, i64* inttoptr (i64 4 to i64*)
  ret void
}

; This is exactly enough to hit the limit of what can be passed
; completely in registers.
; CHECK-LABEL: ret_void_args_i64_i64_i16
define void @ret_void_args_i64_i64_i16(i64 %a, i64 %b, i16 %c) {
  ; CHECK:      sts     5, r9
  ; CHECK-NEXT: sts     4, r8
  store volatile i16 %c, i16* inttoptr (i64 4 to i16*)
  ret void
}
