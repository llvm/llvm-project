; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test that i1 extending loads and truncating stores are assembled properly.

; CHECK-LABEL: load_u_i1_i32:
; CHECK:      i32.load8_u $push[[NUM0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM0]]{{$}}
define i32 @load_u_i1_i32(ptr %p) {
  %v = load i1, ptr %p
  %e = zext i1 %v to i32
  ret i32 %e
}

; CHECK-LABEL: load_s_i1_i32:
; CHECK:      i32.const $push[[NUM3:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.load8_u $push[[NUM0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: i32.const $push[[NUM1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.and $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: i32.sub $push[[NUM4:[0-9]+]]=, $pop[[NUM3]], $pop[[NUM2]]{{$}}
; CHECK-NEXT: return $pop[[NUM4]]{{$}}
define i32 @load_s_i1_i32(ptr %p) {
  %v = load i1, ptr %p
  %e = sext i1 %v to i32
  ret i32 %e
}

; CHECK-LABEL: load_u_i1_i64:
; CHECK:      i64.load8_u $push[[NUM0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM0]]{{$}}
define i64 @load_u_i1_i64(ptr %p) {
  %v = load i1, ptr %p
  %e = zext i1 %v to i64
  ret i64 %e
}

; CHECK-LABEL: load_s_i1_i64:
; CHECK:      i64.const $push[[NUM3:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.load8_u $push[[NUM0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: i64.const $push[[NUM1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.and $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: i64.sub $push[[NUM4:[0-9]+]]=, $pop[[NUM3]], $pop[[NUM2]]{{$}}
; CHECK-NEXT: return $pop[[NUM4]]{{$}}
define i64 @load_s_i1_i64(ptr %p) {
  %v = load i1, ptr %p
  %e = sext i1 %v to i64
  ret i64 %e
}

; CHECK-LABEL: store_i32_i1:
; CHECK:      i32.const $push[[NUM0:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.and $push[[NUM1:[0-9]+]]=, $1, $pop[[NUM0]]{{$}}
; CHECK-NEXT: i32.store8 0($0), $pop[[NUM1]]{{$}}
define void @store_i32_i1(ptr %p, i32 %v) {
  %t = trunc i32 %v to i1
  store i1 %t, ptr %p
  ret void
}

; CHECK-LABEL: store_i64_i1:
; CHECK:      i64.const $push[[NUM0:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.and $push[[NUM1:[0-9]+]]=, $1, $pop[[NUM0]]{{$}}
; CHECK-NEXT: i64.store8 0($0), $pop[[NUM1]]{{$}}
define void @store_i64_i1(ptr %p, i64 %v) {
  %t = trunc i64 %v to i1
  store i1 %t, ptr %p
  ret void
}
