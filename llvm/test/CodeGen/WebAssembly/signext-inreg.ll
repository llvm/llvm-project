; RUN: llc < %s -mattr=+atomics -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals | FileCheck %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals | FileCheck %s --check-prefix=NOATOMIC

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; CHECK-LABEL: i32_extend8_s:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.extend8_s $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}

; NOATOMIC-LABEL: i32_extend8_s
; NOATOMIC-NOT: i32.extend8_s
define i32 @i32_extend8_s(i8 %x) {
  %a = sext i8 %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_extend16_s:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.extend16_s $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}

; NOATOMIC-LABEL: i32_extend16_s
; NOATOMIC-NOT: i32.extend16_s
define i32 @i32_extend16_s(i16 %x) {
  %a = sext i16 %x to i32
  ret i32 %a
}

; CHECK-LABEL: i64_extend8_s:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.extend_u/i32 $push[[NUM1:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: i64.extend8_s $push[[NUM2:[0-9]+]]=, $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}

; NOATOMIC-LABEL: i64_extend8_s
; NOATOMIC-NOT: i64.extend8_s
define i64 @i64_extend8_s(i8 %x) {
  %a = sext i8 %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_extend16_s:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.extend_u/i32 $push[[NUM1:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: i64.extend16_s $push[[NUM2:[0-9]+]]=, $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}

; NOATOMIC-LABEL: i64_extend16_s
; NOATOMIC-NOT: i16.extend16_s
define i64 @i64_extend16_s(i16 %x) {
  %a = sext i16 %x to i64
  ret i64 %a
}

; No SIGN_EXTEND_INREG is needed for 32->64 extension.
; CHECK-LABEL: i64_extend32_s:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: i64.extend_s/i32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_extend32_s(i32 %x) {
  %a = sext i32 %x to i64
  ret i64 %a
}

