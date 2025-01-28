; RUN: llc < %s -mtriple=avr | FileCheck %s

declare void @ret_void_args_struct_i8_i32({ i8, i32 } %a)
declare void @ret_void_args_struct_i8_i8_i8_i8({ i8, i8, i8, i8 } %a)
declare void @ret_void_args_struct_i32_i16_i8({ i32, i16, i8} %a)
declare void @ret_void_args_struct_i8_i32_struct_i32_i8({ i8, i32 } %a, { i32, i8 } %b)

; CHECK-LABEL: call_void_args_struct_i8_i32
define void @call_void_args_struct_i8_i32() {
    ; CHECK: ldi r20, 64
    ; CHECK-NEXT: r21,
    ; CHECK-NEXT: r22,
    ; CHECK-NEXT: r23,
    ; CHECK-NEXT: r24,
    call void @ret_void_args_struct_i8_i32({ i8, i32 } { i8 64, i32 16909060 })
    ret void
}

; CHECK-LABEL: @call_void_args_struct_i8_i8_i8_i8
define void @call_void_args_struct_i8_i8_i8_i8() {
    ; CHECK: ldi r22, 1
    ; CHECK-NEXT: ldi r23, 2
    ; CHECK-NEXT: ldi r24, 3
    ; CHECK-NEXT: ldi r25, 4
    call void @ret_void_args_struct_i8_i8_i8_i8({ i8, i8, i8, i8 } { i8 1, i8 2, i8 3, i8 4 })
    ret void
}

; CHECK-LABEL: @call_void_args_struct_i32_i16_i8
define void @call_void_args_struct_i32_i16_i8() {
    ; CHECK: ldi r18, 4
    ; CHECK-NEXT: ldi r19, 3
    ; CHECK-NEXT: ldi r20, 2
    ; CHECK-NEXT: ldi r21, 1
    ; CHECK-NEXT: ldi r22, 23
    ; CHECK-NEXT: ldi r23, 22
    ; CHECK-NEXT: ldi r24, 64
    call void @ret_void_args_struct_i32_i16_i8({ i32, i16, i8 } { i32 16909060, i16 5655, i8 64 })
    ret void
}

; CHECK-LABEL: @call_void_args_struct_i8_i32_struct_i32_i8
define void @call_void_args_struct_i8_i32_struct_i32_i8() {
    ; CHECK: ldi r20, 64
    ; CHECK: ldi r18, 65
    call void @ret_void_args_struct_i8_i32_struct_i32_i8({ i8, i32 } { i8 64, i32 16909060 }, { i32, i8 } { i32 287454020, i8 65 })
    ret void
}
