; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck --check-prefixes CHECK,CHK32 %s
; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -fast-isel -fast-isel-abort=1 | FileCheck --check-prefixes CHECK,CHK32 %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck --check-prefixes CHECK,CHK64 %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -fast-isel -fast-isel-abort=1 | FileCheck --check-prefixes CHECK,CHK64 %s

; Test that basic loads are assembled properly.

; CHECK-LABEL: ldi32:
; CHK32-NEXT: .functype ldi32 (i32) -> (i32){{$}}
; CHK64-NEXT: .functype ldi32 (i64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.load $push[[NUM:[0-9]+]]=, 0($pop[[L0]]){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ldi32(ptr %p) {
  %v = load i32, ptr %p
  ret i32 %v
}

; CHECK-LABEL: ldi64:
; CHK32-NEXT: .functype ldi64 (i32) -> (i64){{$}}
; CHK64-NEXT: .functype ldi64 (i64) -> (i64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.load $push[[NUM:[0-9]+]]=, 0($pop[[L0]]){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @ldi64(ptr %p) {
  %v = load i64, ptr %p
  ret i64 %v
}

; CHECK-LABEL: ldf32:
; CHK32-NEXT: .functype ldf32 (i32) -> (f32){{$}}
; CHK64-NEXT: .functype ldf32 (i64) -> (f32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: f32.load $push[[NUM:[0-9]+]]=, 0($pop[[L0]]){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @ldf32(ptr %p) {
  %v = load float, ptr %p
  ret float %v
}

; CHECK-LABEL: ldf64:
; CHK32-NEXT: .functype ldf64 (i32) -> (f64){{$}}
; CHK64-NEXT: .functype ldf64 (i64) -> (f64){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: f64.load $push[[NUM:[0-9]+]]=, 0($pop[[L0]]){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @ldf64(ptr %p) {
  %v = load double, ptr %p
  ret double %v
}
