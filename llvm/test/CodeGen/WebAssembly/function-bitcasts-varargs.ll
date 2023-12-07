; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s

; Test that function pointer casts casting away varargs are replaced with
; wrappers.

target triple = "wasm32-unknown-unknown"

define void @callWithArgs() {
entry:
  call void @underspecified(i32 0, i32 1)
  call void(...) @specified(i32 0, i32 1)
  ret void
}

declare void @underspecified(...)
declare void @specified(i32, i32)

; CHECK: callWithArgs:
; CHECK: i32.const	$push1=, 0
; CHECK-NEXT: i32.const	$push0=, 1
; CHECK-NEXT: call    	.Lunderspecified_bitcast, $pop1, $pop0
; CHECK: call    	.Lspecified_bitcast, $pop{{[0-9]+$}}

; CHECK: .Lunderspecified_bitcast:
; CHECK-NEXT: .functype .Lunderspecified_bitcast (i32, i32) -> (){{$}}
; CHECK: call    	underspecified, $pop{{[0-9]+$}}

; CHECK: .Lspecified_bitcast:
; CHECK-NEXT: .functype .Lspecified_bitcast (i32) -> (){{$}}
; CHECK: call    	specified, $pop{{[0-9]+}}, $pop{{[0-9]+$}}
