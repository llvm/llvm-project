; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test zeroext and signext ABI keywords

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: z2s_func:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32{{$}}
; CHECK-NEXT: i32.const $[[NUM0:[0-9]+]], 24{{$}}
; CHECK-NEXT: i32.shl $push[[NUM2:[0-9]+]], $0, $[[NUM0]]{{$}}
; CHECK-NEXT: i32.shr_s $push[[NUM3:[0-9]+]], $pop[[NUM2]], $[[NUM0]]{{$}}
; CHECK-NEXT: return $pop[[NUM3]]{{$}}
define signext i8 @z2s_func(i8 zeroext %t) {
  ret i8 %t
}

; CHECK-LABEL: s2z_func:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.const $push[[NUM0:[0-9]+]], 255{{$}}
; CHECK-NEXT: i32.and $push[[NUM1:[0-9]+]], $0, $pop[[NUM0]]{{$}}
; CHECK-NEXT: return $pop[[NUM1]]{{$}}
define zeroext i8 @s2z_func(i8 signext %t) {
  ret i8 %t
}

; CHECK-LABEL: z2s_call:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result i32
; CHECK-NEXT: i32.const $push[[NUM0:[0-9]+]], 255{{$}}
; CHECK-NEXT: i32.and $push[[NUM1:[0-9]+]], $0, $pop[[NUM0]]{{$}}
; CHECK-NEXT: call z2s_func, $push[[NUM2:[0-9]+]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @z2s_call(i32 %t) {
  %s = trunc i32 %t to i8
  %u = call signext i8 @z2s_func(i8 zeroext %s)
  %v = sext i8 %u to i32
  ret i32 %v
}

; CHECK-LABEL: s2z_call:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result i32
; CHECK-NEXT: .local i32{{$}}
; CHECK-NEXT: i32.const $[[NUM0:[0-9]+]], 24{{$}}
; CHECK-NEXT: i32.shl $push[[NUM1:[0-9]+]], $0, $[[NUM0]]{{$}}
; CHECK-NEXT: i32.shr_s $push[[NUM2:[0-9]+]], $pop[[NUM1]], $[[NUM0]]{{$}}
; CHECK-NEXT: call s2z_func, $push[[NUM3:[0-9]]], $pop[[NUM2]]{{$}}
; CHECK-NEXT: i32.shl $push[[NUM4:[0-9]+]], $pop[[NUM3]], $[[NUM0]]{{$}}
; CHECK-NEXT: i32.shr_s $push[[NUM5:[0-9]+]], $pop[[NUM4]], $[[NUM0]]{{$}}
; CHECK-NEXT: return $pop[[NUM5]]{{$}}
define i32 @s2z_call(i32 %t) {
  %s = trunc i32 %t to i8
  %u = call zeroext i8 @s2z_func(i8 signext %s)
  %v = sext i8 %u to i32
  ret i32 %v
}
