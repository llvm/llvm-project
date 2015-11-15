; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test zeroext and signext ABI keywords

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: z2s_func:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32, i32, i32, i32{{$}}
; CHECK-NEXT: i32.const $push, 24{{$}}
; CHECK-NEXT: set_local 1, $pop{{$}}
; CHECK-NEXT: i32.shl $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
; CHECK-NEXT: i32.shr_s $push, (get_local 2), (get_local 1){{$}}
; CHECK-NEXT: set_local 3, $pop{{$}}
; CHECK-NEXT: return (get_local 3){{$}}
define signext i8 @z2s_func(i8 zeroext %t) {
  ret i8 %t
}

; CHECK-LABEL: s2z_func:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32, i32, i32{{$}}
; CHECK-NEXT: i32.const $push, 255{{$}}
; CHECK-NEXT: set_local 1, $pop{{$}}
; CHECK-NEXT: i32.and $push, (get_local 0), (get_local 1){{$}}
; CHECK-NEXT: set_local 2, $pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define zeroext i8 @s2z_func(i8 signext %t) {
  ret i8 %t
}

; CHECK-LABEL: z2s_call:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result i32
; CHECK-NEXT: .local i32, i32, i32, i32
; CHECK-NEXT: i32.const $push, 255
; CHECK-NEXT: set_local 1, $pop
; CHECK-NEXT: i32.and $push, (get_local 0), (get_local 1)
; CHECK-NEXT: set_local 2, $pop
; CHECK-NEXT: call z2s_func, $push, (get_local 2)
; CHECK-NEXT: set_local 3, $pop
; CHECK-NEXT: return (get_local 3)
define i32 @z2s_call(i32 %t) {
  %s = trunc i32 %t to i8
  %u = call signext i8 @z2s_func(i8 zeroext %s)
  %v = sext i8 %u to i32
  ret i32 %v
}

; CHECK-LABEL: s2z_call:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result i32
; CHECK-NEXT: .local i32, i32, i32, i32, i32, i32, i32
; CHECK-NEXT: i32.const $push, 24
; CHECK-NEXT: set_local 1, $pop
; CHECK-NEXT: i32.shl $push, (get_local 0), (get_local 1)
; CHECK-NEXT: set_local 2, $pop
; CHECK-NEXT: i32.shr_s $push, (get_local 2), (get_local 1)
; CHECK-NEXT: set_local 3, $pop
; CHECK-NEXT: call s2z_func, $push, (get_local 3)
; CHECK-NEXT: set_local 4, $pop
; CHECK-NEXT: i32.shl $push, (get_local 4), (get_local 1)
; CHECK-NEXT: set_local 5, $pop
; CHECK-NEXT: i32.shr_s $push, (get_local 5), (get_local 1)
; CHECK-NEXT: set_local 6, $pop
; CHECK-NEXT: return (get_local 6)
define i32 @s2z_call(i32 %t) {
  %s = trunc i32 %t to i8
  %u = call zeroext i8 @s2z_func(i8 signext %s)
  %v = sext i8 %u to i32
  ret i32 %v
}
