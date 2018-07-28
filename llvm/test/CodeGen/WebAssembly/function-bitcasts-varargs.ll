; RUN: llc < %s -asm-verbose=false -wasm-temporary-workarounds=false | FileCheck %s

; Test that function pointer casts casting away varargs are replaced with
; wrappers.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @callWithArgs() {
entry:
  call void bitcast (void (...)* @underspecified to void (i32, i32)*)(i32 0, i32 1)
  call void(...) bitcast (void (i32, i32)* @specified to void (...)*)(i32 0, i32 1)
  ret void
}

declare void @underspecified(...)
declare void @specified(i32, i32)

; CHECK: callWithArgs:
; CHECK: i32.const	$push1=, 0
; CHECK-NEXT: i32.const	$push0=, 1
; CHECK-NEXT: call    	.Lbitcast@FUNCTION, $pop1, $pop0
; CHECK: call    	.Lbitcast.1@FUNCTION, $pop{{[0-9]+$}}

; CHECK: .Lbitcast:
; CHECK-NEXT: .param  	i32, i32{{$}}
; CHECK: call    	underspecified@FUNCTION, $pop{{[0-9]+$}}

; CHECK: .Lbitcast.1:
; CHECK-NEXT: .param  	i32{{$}}
; CHECK: call    	specified@FUNCTION, $pop{{[0-9]+}}, $pop{{[0-9]+$}}
