; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 | FileCheck %s

; Tests that redundant masking and conversions are folded out
; following SIMD reduction instructions.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================
declare i32 @llvm.wasm.anytrue.v16i8(<16 x i8>)
declare i32 @llvm.wasm.alltrue.v16i8(<16 x i8>)

; CHECK-LABEL: any_v16i8_trunc:
; CHECK-NEXT: .functype any_v16i8_trunc (v128) -> (i32){{$}}
; CHECK-NEXT: i8x16.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v16i8_trunc(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.anytrue.v16i8(<16 x i8> %x)
  %b = trunc i32 %a to i1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: any_v16i8_ne:
; CHECK-NEXT: .functype any_v16i8_ne (v128) -> (i32){{$}}
; CHECK-NEXT: i8x16.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v16i8_ne(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.anytrue.v16i8(<16 x i8> %x)
  %b = icmp ne i32 %a, 0
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: any_v16i8_eq:
; CHECK-NEXT: .functype any_v16i8_eq (v128) -> (i32){{$}}
; CHECK-NEXT: i8x16.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v16i8_eq(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.anytrue.v16i8(<16 x i8> %x)
  %b = icmp eq i32 %a, 1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v16i8_trunc:
; CHECK-NEXT: .functype all_v16i8_trunc (v128) -> (i32){{$}}
; CHECK-NEXT: i8x16.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v16i8_trunc(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.alltrue.v16i8(<16 x i8> %x)
  %b = trunc i32 %a to i1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v16i8_ne:
; CHECK-NEXT: .functype all_v16i8_ne (v128) -> (i32){{$}}
; CHECK-NEXT: i8x16.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v16i8_ne(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.alltrue.v16i8(<16 x i8> %x)
  %b = icmp ne i32 %a, 0
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v16i8_eq:
; CHECK-NEXT: .functype all_v16i8_eq (v128) -> (i32){{$}}
; CHECK-NEXT: i8x16.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v16i8_eq(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.alltrue.v16i8(<16 x i8> %x)
  %b = icmp eq i32 %a, 1
  %c = zext i1 %b to i32
  ret i32 %c
}

; ==============================================================================
; 8 x i16
; ==============================================================================
declare i32 @llvm.wasm.anytrue.v8i16(<8 x i16>)
declare i32 @llvm.wasm.alltrue.v8i16(<8 x i16>)

; CHECK-LABEL: any_v8i16_trunc:
; CHECK-NEXT: .functype any_v8i16_trunc (v128) -> (i32){{$}}
; CHECK-NEXT: i16x8.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v8i16_trunc(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.anytrue.v8i16(<8 x i16> %x)
  %b = trunc i32 %a to i1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: any_v8i16_ne:
; CHECK-NEXT: .functype any_v8i16_ne (v128) -> (i32){{$}}
; CHECK-NEXT: i16x8.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v8i16_ne(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.anytrue.v8i16(<8 x i16> %x)
  %b = icmp ne i32 %a, 0
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: any_v8i16_eq:
; CHECK-NEXT: .functype any_v8i16_eq (v128) -> (i32){{$}}
; CHECK-NEXT: i16x8.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v8i16_eq(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.anytrue.v8i16(<8 x i16> %x)
  %b = icmp eq i32 %a, 1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v8i16_trunc:
; CHECK-NEXT: .functype all_v8i16_trunc (v128) -> (i32){{$}}
; CHECK-NEXT: i16x8.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v8i16_trunc(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.alltrue.v8i16(<8 x i16> %x)
  %b = trunc i32 %a to i1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v8i16_ne:
; CHECK-NEXT: .functype all_v8i16_ne (v128) -> (i32){{$}}
; CHECK-NEXT: i16x8.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v8i16_ne(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.alltrue.v8i16(<8 x i16> %x)
  %b = icmp ne i32 %a, 0
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v8i16_eq:
; CHECK-NEXT: .functype all_v8i16_eq (v128) -> (i32){{$}}
; CHECK-NEXT: i16x8.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v8i16_eq(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.alltrue.v8i16(<8 x i16> %x)
  %b = icmp eq i32 %a, 1
  %c = zext i1 %b to i32
  ret i32 %c
}

; ==============================================================================
; 4 x i32
; ==============================================================================
declare i32 @llvm.wasm.anytrue.v4i32(<4 x i32>)
declare i32 @llvm.wasm.alltrue.v4i32(<4 x i32>)

; CHECK-LABEL: any_v4i32_trunc:
; CHECK-NEXT: .functype any_v4i32_trunc (v128) -> (i32){{$}}
; CHECK-NEXT: i32x4.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v4i32_trunc(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.anytrue.v4i32(<4 x i32> %x)
  %b = trunc i32 %a to i1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: any_v4i32_ne:
; CHECK-NEXT: .functype any_v4i32_ne (v128) -> (i32){{$}}
; CHECK-NEXT: i32x4.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v4i32_ne(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.anytrue.v4i32(<4 x i32> %x)
  %b = icmp ne i32 %a, 0
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: any_v4i32_eq:
; CHECK-NEXT: .functype any_v4i32_eq (v128) -> (i32){{$}}
; CHECK-NEXT: i32x4.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v4i32_eq(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.anytrue.v4i32(<4 x i32> %x)
  %b = icmp eq i32 %a, 1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v4i32_trunc:
; CHECK-NEXT: .functype all_v4i32_trunc (v128) -> (i32){{$}}
; CHECK-NEXT: i32x4.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v4i32_trunc(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.alltrue.v4i32(<4 x i32> %x)
  %b = trunc i32 %a to i1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v4i32_ne:
; CHECK-NEXT: .functype all_v4i32_ne (v128) -> (i32){{$}}
; CHECK-NEXT: i32x4.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v4i32_ne(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.alltrue.v4i32(<4 x i32> %x)
  %b = icmp ne i32 %a, 0
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v4i32_eq:
; CHECK-NEXT: .functype all_v4i32_eq (v128) -> (i32){{$}}
; CHECK-NEXT: i32x4.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v4i32_eq(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.alltrue.v4i32(<4 x i32> %x)
  %b = icmp eq i32 %a, 1
  %c = zext i1 %b to i32
  ret i32 %c
}

; ==============================================================================
; 2 x i64
; ==============================================================================
declare i32 @llvm.wasm.anytrue.v2i64(<2 x i64>)
declare i32 @llvm.wasm.alltrue.v2i64(<2 x i64>)

; CHECK-LABEL: any_v2i64_trunc:
; CHECK-NEXT: .functype any_v2i64_trunc (v128) -> (i32){{$}}
; CHECK-NEXT: i64x2.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v2i64_trunc(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.anytrue.v2i64(<2 x i64> %x)
  %b = trunc i32 %a to i1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: any_v2i64_ne:
; CHECK-NEXT: .functype any_v2i64_ne (v128) -> (i32){{$}}
; CHECK-NEXT: i64x2.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v2i64_ne(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.anytrue.v2i64(<2 x i64> %x)
  %b = icmp ne i32 %a, 0
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: any_v2i64_eq:
; CHECK-NEXT: .functype any_v2i64_eq (v128) -> (i32){{$}}
; CHECK-NEXT: i64x2.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @any_v2i64_eq(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.anytrue.v2i64(<2 x i64> %x)
  %b = icmp eq i32 %a, 1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v2i64_trunc:
; CHECK-NEXT: .functype all_v2i64_trunc (v128) -> (i32){{$}}
; CHECK-NEXT: i64x2.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v2i64_trunc(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.alltrue.v2i64(<2 x i64> %x)
  %b = trunc i32 %a to i1
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v2i64_ne:
; CHECK-NEXT: .functype all_v2i64_ne (v128) -> (i32){{$}}
; CHECK-NEXT: i64x2.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v2i64_ne(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.alltrue.v2i64(<2 x i64> %x)
  %b = icmp ne i32 %a, 0
  %c = zext i1 %b to i32
  ret i32 %c
}

; CHECK-LABEL: all_v2i64_eq:
; CHECK-NEXT: .functype all_v2i64_eq (v128) -> (i32){{$}}
; CHECK-NEXT: i64x2.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @all_v2i64_eq(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.alltrue.v2i64(<2 x i64> %x)
  %b = icmp eq i32 %a, 1
  %c = zext i1 %b to i32
  ret i32 %c
}
