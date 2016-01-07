; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that constant offsets can be folded into global addresses.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; FIXME: make this 'external' and make sure it still works. WebAssembly
;        currently only supports linking single files, so 'external' makes
;        little sense.
@x = global [0 x i32] zeroinitializer
@y = global [50 x i32] zeroinitializer

; Test basic constant offsets of both defined and external symbols.

; CHECK-LABEL: test0:
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.const $push0=, x+188{{$}}
; CHECK=NEXT: return $pop0{{$}}
define i32* @test0() {
  ret i32* getelementptr ([0 x i32], [0 x i32]* @x, i32 0, i32 47)
}

; CHECK-LABEL: test1:
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.const $push0=, y+188{{$}}
; CHECK=NEXT: return $pop0{{$}}
define i32* @test1() {
  ret i32* getelementptr ([50 x i32], [50 x i32]* @y, i32 0, i32 47)
}

; Test zero offsets.

; CHECK-LABEL: test2:
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.const $push0=, x{{$}}
; CHECK=NEXT: return $pop0{{$}}
define i32* @test2() {
  ret i32* getelementptr ([0 x i32], [0 x i32]* @x, i32 0, i32 0)
}

; CHECK-LABEL: test3:
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: i32.const $push0=, y{{$}}
; CHECK=NEXT: return $pop0{{$}}
define i32* @test3() {
  ret i32* getelementptr ([50 x i32], [50 x i32]* @y, i32 0, i32 0)
}
