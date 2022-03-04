; RUN: llc -mtriple=avr < %s | FileCheck %s

; CHECK-LABEL: ret_void_args_struct_i8_i32
define void @ret_void_args_struct_i8_i32({ i8, i32 } %a) {
start:
  ; CHECK:      sts     4, r20
  %0 = extractvalue { i8, i32 } %a, 0
  store volatile i8 %0, i8* inttoptr (i64 4 to i8*)

  ; CHECK-NEXT: sts     8, r24
  ; CHECK-NEXT: sts     7, r23
  ; CHECK-NEXT: sts     6, r22
  ; CHECK-NEXT: sts     5, r21
  %1 = extractvalue { i8, i32 } %a, 1
  store volatile i32 %1, i32* inttoptr (i64 5 to i32*)
  ret void
}

; CHECK-LABEL: ret_void_args_struct_i8_i8_i8_i8
define void @ret_void_args_struct_i8_i8_i8_i8({ i8, i8, i8, i8 } %a) {
start:
  ; CHECK:      sts     4, r22
  %0 = extractvalue { i8, i8, i8, i8 } %a, 0
  store volatile i8 %0, i8* inttoptr (i64 4 to i8*)
  ; CHECK-NEXT: sts     5, r23
  %1 = extractvalue { i8, i8, i8, i8 } %a, 1
  store volatile i8 %1, i8* inttoptr (i64 5 to i8*)
  ; CHECK-NEXT: sts     6, r24
  %2 = extractvalue { i8, i8, i8, i8 } %a, 2
  store volatile i8 %2, i8* inttoptr (i64 6 to i8*)
  ; CHECK-NEXT: sts     7, r25
  %3 = extractvalue { i8, i8, i8, i8 } %a, 3
  store volatile i8 %3, i8* inttoptr (i64 7 to i8*)
  ret void
}

; CHECK-LABEL: ret_void_args_struct_i32_16_i8
define void @ret_void_args_struct_i32_16_i8({ i32, i16, i8} %a) {
start:
  ; CHECK:      sts     7, r21
  ; CHECK-NEXT: sts     6, r20
  ; CHECK-NEXT: sts     5, r19
  ; CHECK-NEXT: sts     4, r18
  %0 = extractvalue { i32, i16, i8 } %a, 0
  store volatile i32 %0, i32* inttoptr (i64 4 to i32*)

  ; CHECK-NEXT: sts     5, r23
  ; CHECK-NEXT: sts     4, r22
  %1 = extractvalue { i32, i16, i8 } %a, 1
  store volatile i16 %1, i16* inttoptr (i64 4 to i16*)

  ; CHECK-NEXT: sts     4, r24
  %2 = extractvalue { i32, i16, i8 } %a, 2
  store volatile i8 %2, i8* inttoptr (i64 4 to i8*)
  ret void
}

; CHECK-LABEL: ret_void_args_struct_i8_i32_struct_i32_i8
define void @ret_void_args_struct_i8_i32_struct_i32_i8({ i8, i32 } %a, { i32, i8 } %b) {
start:
  ; CHECK:      sts     4, r20
  %0 = extractvalue { i8, i32 } %a, 0
  store volatile i8 %0, i8* inttoptr (i64 4 to i8*)

  ; CHECK-NEXT: sts     8, r24
  ; CHECK-NEXT: sts     7, r23
  ; CHECK-NEXT: sts     6, r22
  ; CHECK-NEXT: sts     5, r21
  %1 = extractvalue { i8, i32 } %a, 1
  store volatile i32 %1, i32* inttoptr (i64 5 to i32*)

  ; CHECK-NEXT:      sts     9, r17
  ; CHECK-NEXT:      sts     8, r16
  ; CHECK-NEXT:      sts     7, r15
  ; CHECK-NEXT:      sts     6, r14
  %2 = extractvalue { i32, i8 } %b, 0
  store volatile i32 %2, i32* inttoptr (i64 6 to i32*)

  ; CHECK-NEXT: sts     7, r18
  %3 = extractvalue { i32, i8 } %b, 1
  store volatile i8 %3, i8* inttoptr (i64 7 to i8*)
  ret void
}

; NOTE: The %0 (8-byte array) costs 8 registers and %1 (10-byte array)
; NOTE: costs 10 registers.
define i8 @foo0([8 x i8] %0, [10 x i8] %1) {
; CHECK-LABEL: foo0:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    sub r18, r8
; CHECK-NEXT:    mov r24, r18
; CHECK-NEXT:    ret
  %3 = extractvalue [8 x i8] %0, 0
  %4 = extractvalue [10 x i8] %1, 0
  %5 = sub i8 %3, %4
  ret i8 %5
}

; NOTE: The %0 (7-byte array) costs 8 registers and %1 (9-byte array)
; NOTE: costs 10 registers.
define i8 @foo1([7 x i8] %0, [9 x i8] %1) {
; CHECK-LABEL: foo1:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    sub r18, r8
; CHECK-NEXT:    mov r24, r18
; CHECK-NEXT:    ret
  %3 = extractvalue [7 x i8] %0, 0
  %4 = extractvalue [9 x i8] %1, 0
  %5 = sub i8 %3, %4
  ret i8 %5
}

; NOTE: Each argument (6-byte array) costs 6 registers.
define i8 @foo2([6 x i8] %0, [6 x i8] %1, [6 x i8] %2) {
; CHECK-LABEL: foo2:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    sub r20, r14
; CHECK-NEXT:    add r20, r8
; CHECK-NEXT:    mov r24, r20
; CHECK-NEXT:    ret
  %4 = extractvalue [6 x i8] %0, 0
  %5 = extractvalue [6 x i8] %1, 0
  %6 = extractvalue [6 x i8] %2, 0
  %7 = sub i8 %4, %5
  %8 = add i8 %7, %6
  ret i8 %8
}

; NOTE: The %0 (9-byte array) costs 10 registers. Though there are
; NOTE: 8 registers are vacant, the %b (9-byte array) has to be dropped
; NOTE: to the stack.
define i8 @foo3([9 x i8] %0, [9 x i8] %1) {
; CHECK-LABEL: foo3:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    push r16
; CHECK-NEXT:    push r28
; CHECK-NEXT:    push r29
; CHECK-NEXT:    in r28, 61
; CHECK-NEXT:    in r29, 62
; CHECK-NEXT:    ldd r24, Y+6
; CHECK-NEXT:    sub r16, r24
; CHECK-NEXT:    mov r24, r16
; CHECK-NEXT:    pop r29
; CHECK-NEXT:    pop r28
; CHECK-NEXT:    pop r16
; CHECK-NEXT:    ret
  %3 = extractvalue [9 x i8] %0, 0
  %4 = extractvalue [9 x i8] %1, 0
  %5 = sub i8 %3, %4
  ret i8 %5
}

; NOTE: Both %0 and %1 are 7-byte arrays, and cost total 16 registers.
; NOTE: Though there are 2 registers are vacant, the %2 (7-byte array) has to
; NOTE: be dropped to the stack.
define i8 @foo4([7 x i8] %0, [7 x i8] %1, [7 x i8] %2) {
; CHECK-LABEL: foo4:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    push r28
; CHECK-NEXT:    push r29
; CHECK-NEXT:    in r28, 61
; CHECK-NEXT:    in r29, 62
; CHECK-NEXT:    sub r18, r10
; CHECK-NEXT:    ldd r24, Y+5
; CHECK-NEXT:    add r24, r18
; CHECK-NEXT:    pop r29
; CHECK-NEXT:    pop r28
; CHECK-NEXT:    ret
  %4 = extractvalue [7 x i8] %0, 0
  %5 = extractvalue [7 x i8] %1, 0
  %6 = extractvalue [7 x i8] %2, 0
  %7 = sub i8 %4, %5
  %8 = add i8 %7, %6
  ret i8 %8
}
