; RUN: llc < %s -asm-verbose=false -wasm-explicit-locals-codegen-test-mode | FileCheck %s

; Test that function pointer casts that require conversions of arguments or
; return types are converted to unreachable.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @has_i64_arg(i64)
declare i32 @has_ptr_arg(i8*)

define void @test_invalid_rtn() {
entry:
  call i32 bitcast (i32 (i64)* @has_i64_arg to i32 (i32)*)(i32 0)
  ret void
}
; CHECK-LABEL: test_invalid_rtn:
; CHECK-NEXT: i32.const   $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.call $push1=, .Lhas_i64_arg_bitcast_invalid@FUNCTION, $pop[[L0]]{{$}}
; CHECK-NEXT: drop $pop1
; CHECK-NEXT: end_function

define void @test_invalid_arg() {
entry:
  call i32 bitcast (i32 (i8*)* @has_ptr_arg to i32 (i8)*)(i8 2)
  call i32 bitcast (i32 (i8*)* @has_ptr_arg to i32 (i32)*)(i32 2)
  call i32 bitcast (i32 (i8*)* @has_ptr_arg to i32 (i64)*)(i64 3)
  ret void
}

; CHECK-LABEL: test_invalid_arg:
; CHECK-NEXT: 	i32.const	$push[[L0:[0-9]+]]=, 2{{$}}
; CHECK-NEXT: 	i32.call	$push[[L1:[0-9]+]]=, .Lhas_ptr_arg_bitcast_invalid.1@FUNCTION, $pop[[L0]]{{$}}
; CHECK-NEXT: 	drop	$pop[[L1]]{{$}}
; CHECK-NEXT: 	i32.const	$push[[L0:[0-9]+]]=, 2{{$}}
; CHECK-NEXT: 	i32.call	$push[[L1:[0-9]+]]=, has_ptr_arg@FUNCTION, $pop[[L0]]{{$}}
; CHECK-NEXT: 	drop	$pop[[L1]]{{$}}
; CHECK-NEXT: 	i64.const	$push[[L0:[0-9]+]]=, 3{{$}}
; CHECK-NEXT: 	i32.call	$push[[L1:[0-9]+]]=, .Lhas_ptr_arg_bitcast_invalid@FUNCTION, $pop[[L0]]{{$}}
; CHECK-NEXT: 	drop	$pop[[L1]]{{$}}
; CHECK-NEXT: 	end_function

; CHECK-LABEL: .Lhas_i64_arg_bitcast_invalid:
; CHECK-NEXT:  .param  	i32
; CHECK-NEXT:  .result 	i32
; CHECK-NEXT:  unreachable
; CHECK-NEXT:  end_function

; CHECK-LABEL: .Lhas_ptr_arg_bitcast_invalid:
; CHECK-NEXT: 	.param  	i64
; CHECK-NEXT: 	.result 	i32
; CHECK-NEXT: 	unreachable
; CHECK-NEXT: 	end_function

; CHECK-LABEL: .Lhas_ptr_arg_bitcast_invalid.1:
; CHECK-NEXT: 	.param  	i32
; CHECK-NEXT: 	.result 	i32
; CHECK-NEXT: 	unreachable
; CHECK-NEXT: 	end_function
