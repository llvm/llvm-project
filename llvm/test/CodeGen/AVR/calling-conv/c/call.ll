; RUN: llc < %s -march=avr | FileCheck %s

declare void @ret_void_args_i8(i8 %a)
declare void @ret_void_args_i8_i32(i8 %a, i32 %b)
declare void @ret_void_args_i8_i8_i8_i8(i8 %a, i8 %b, i8 %c, i8 %d)
declare void @ret_void_args_i32_i16_i8(i32 %a, i16 %b, i8 %c)
declare void @ret_void_args_i64(i64 %a)
declare void @ret_void_args_i64_i64(i64 %a, i64 %b)
declare void @ret_void_args_i64_i64_i16(i64 %a, i64 %b, i16 %c)

; CHECK-LABEL: call_void_args_i8
define void @call_void_args_i8() {
    ; CHECK: ldi r24, 64
    call void @ret_void_args_i8 (i8 64)
    ret void
}

; CHECK-LABEL: call_void_args_i8_i32
define void @call_void_args_i8_i32() {
    ; CHECK: ldi r20, 4
    ; CHECK-NEXT: ldi r21, 3
    ; CHECK-NEXT: ldi r22, 2
    ; CHECK-NEXT: ldi r23, 1
    ; CHECK-NEXT: ldi r24, 64
    call void @ret_void_args_i8_i32 (i8 64, i32 16909060)
    ret void
}

; CHECK-LABEL: call_void_args_i8_i8_i8_i8
define void @call_void_args_i8_i8_i8_i8() {
    ; CHECK: ldi r24, 1
    ; CHECK-NEXT: ldi r22, 2
    ; CHECK-NEXT: ldi r20, 3
    ; CHECK-NEXT: ldi r18, 4
    call void @ret_void_args_i8_i8_i8_i8(i8 1, i8 2, i8 3, i8 4)
    ret void
}

; CHECK-LABEL: call_void_args_i32_i16_i8
define void @call_void_args_i32_i16_i8() {
    ; CHECK: ldi r22, 4
    ; CHECK-NEXT: ldi r23, 3
    ; CHECK-NEXT: ldi r24, 2
    ; CHECK-NEXT: ldi r25, 1
    ; CHECK-NEXT: ldi r20, 1
    ; CHECK-NEXT: ldi r21, 4
    ; CHECK-NEXT: ldi r18, 64
    call void @ret_void_args_i32_i16_i8(i32 16909060, i16 1025, i8 64)
    ret void
}

; CHECK-LABEL: call_void_args_i64
define void @call_void_args_i64() {
    ; CHECK: ldi r18, 8
    ; CHECK-NEXT: ldi r19, 7
    ; CHECK-NEXT: ldi r20, 6
    ; CHECK-NEXT: ldi r21, 5
    ; CHECK-NEXT: ldi r22, 4
    ; CHECK-NEXT: ldi r23, 3
    ; CHECK-NEXT: ldi r24, 2
    ; CHECK-NEXT: ldi r25, 1
    call void @ret_void_args_i64(i64 72623859790382856)
    ret void
}

; CHECK-LABEL: call_void_args_i64_i64
define void @call_void_args_i64_i64() {
    ; CHECK: ldi r18, 8
    ; CHECK-NEXT: ldi r19, 7
    ; CHECK-NEXT: ldi r20, 6
    ; CHECK-NEXT: ldi r21, 5
    ; CHECK-NEXT: ldi r22, 4
    ; CHECK-NEXT: ldi r23, 3
    ; CHECK-NEXT: ldi r24, 2
    ; CHECK-NEXT: ldi r25, 1
    ; the second arg is in r10:r17, but unordered
    ; CHECK: r17,
    ; CHECK: r10,
    call void @ret_void_args_i64_i64(i64 72623859790382856, i64 651345242494996224)
    ret void
}

; CHECK-LABEL: call_void_args_i64_i64_i16
define void @call_void_args_i64_i64_i16() {
    ; CHECK: r8,
    ; CHECK: r9,
    call void @ret_void_args_i64_i64_i16(i64 72623859790382856, i64 651345242494996224, i16 5655)
    ret void
}
