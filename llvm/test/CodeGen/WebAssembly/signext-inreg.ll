; RUN: llc < %s -mattr=+sign-ext -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s --check-prefix=NOSIGNEXT

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: i32_extend8_s:
; CHECK-NEXT: .functype i32_extend8_s (i32) -> (i32){{$}}
; CHECK-NEXT: i32.extend8_s $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}

; NOSIGNEXT-LABEL: i32_extend8_s
; NOSIGNEXT-NOT: i32.extend8_s
define i32 @i32_extend8_s(i8 %x) {
  %a = sext i8 %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_extend16_s:
; CHECK-NEXT: .functype i32_extend16_s (i32) -> (i32){{$}}
; CHECK-NEXT: i32.extend16_s $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}

; NOSIGNEXT-LABEL: i32_extend16_s
; NOSIGNEXT-NOT: i32.extend16_s
define i32 @i32_extend16_s(i16 %x) {
  %a = sext i16 %x to i32
  ret i32 %a
}

; CHECK-LABEL: i64_extend8_s:
; CHECK-NEXT: .functype i64_extend8_s (i32) -> (i64){{$}}
; CHECK-NEXT: i64.extend_u/i32 $push[[NUM1:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: i64.extend8_s $push[[NUM2:[0-9]+]]=, $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}

; NOSIGNEXT-LABEL: i64_extend8_s
; NOSIGNEXT-NOT: i64.extend8_s
define i64 @i64_extend8_s(i8 %x) {
  %a = sext i8 %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_extend16_s:
; CHECK-NEXT: .functype i64_extend16_s (i32) -> (i64){{$}}
; CHECK-NEXT: i64.extend_u/i32 $push[[NUM1:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: i64.extend16_s $push[[NUM2:[0-9]+]]=, $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}

; NOSIGNEXT-LABEL: i64_extend16_s
; NOSIGNEXT-NOT: i16.extend16_s
define i64 @i64_extend16_s(i16 %x) {
  %a = sext i16 %x to i64
  ret i64 %a
}

; No SIGN_EXTEND_INREG is needed for 32->64 extension.
; CHECK-LABEL: i64_extend32_s:
; CHECK-NEXT: .functype i64_extend32_s (i32) -> (i64){{$}}
; CHECK-NEXT: i64.extend_s/i32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_extend32_s(i32 %x) {
  %a = sext i32 %x to i64
  ret i64 %a
}

