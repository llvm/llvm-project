; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s

; Test that function pointer casts that require conversions of arguments or
; return types are converted to unreachable.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @has_i64_arg(i64)
declare i32 @has_ptr_arg(i8*)

; CHECK-LABEL: test_invalid_rtn:
; CHECK:      i32.const   $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.call $push[[L1:[0-9]+]]=, .Lhas_i64_arg_bitcast_invalid.2, $pop[[L0]]{{$}}
; CHECK-NEXT: drop $pop[[L1]]{{$}}
; CHECK-NEXT: i64.const   $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.call $push[[L1:[0-9]+]]=, .Lhas_i64_arg_bitcast_invalid, $pop[[L0]]{{$}}
; CHECK-NEXT: drop $pop[[L1]]{{$}}
; CHECK-NEXT: end_function
define void @test_invalid_rtn() {
entry:
  call i32 bitcast (i32 (i64)* @has_i64_arg to i32 (i32)*)(i32 0)
  call [1 x i64] bitcast (i32 (i64)* @has_i64_arg to [1 x i64] (i64)*)(i64 0)
  ret void
}

; CHECK-LABEL: test_struct_rtn:
; CHECK: 	call    	has_i64_arg, $pop6, $pop0
define void @test_struct_rtn() {
  call {i32, i32} bitcast (i32 (i64)* @has_i64_arg to {i32, i32} (i64)*)(i64 0)
  ret void
}

; CHECK-LABEL: test_invalid_arg:
; CHECK:      	i32.const	$push[[L0:[0-9]+]]=, 2{{$}}
; CHECK-NEXT: 	i32.call	$push[[L1:[0-9]+]]=, .Lhas_ptr_arg_bitcast_invalid.4, $pop[[L0]]{{$}}
; CHECK-NEXT: 	drop	$pop[[L1]]{{$}}
; CHECK-NEXT: 	i32.const	$push[[L0:[0-9]+]]=, 2{{$}}
; CHECK-NEXT: 	i32.call	$push[[L1:[0-9]+]]=, has_ptr_arg, $pop[[L0]]{{$}}
; CHECK-NEXT: 	drop	$pop[[L1]]{{$}}
; CHECK-NEXT: 	i64.const	$push[[L0:[0-9]+]]=, 3{{$}}
; CHECK-NEXT: 	i32.call	$push[[L1:[0-9]+]]=, .Lhas_ptr_arg_bitcast_invalid, $pop[[L0]]{{$}}
; CHECK-NEXT: 	drop	$pop[[L1]]{{$}}
; CHECK-NEXT: 	end_function
define void @test_invalid_arg() {
entry:
  call i32 bitcast (i32 (i8*)* @has_ptr_arg to i32 (i8)*)(i8 2)
  call i32 bitcast (i32 (i8*)* @has_ptr_arg to i32 (i32)*)(i32 2)
  call i32 bitcast (i32 (i8*)* @has_ptr_arg to i32 (i64)*)(i64 3)
  ret void
}

; CHECK-LABEL: .Lhas_i64_arg_bitcast_invalid:
; CHECK-NEXT:  .functype .Lhas_i64_arg_bitcast_invalid (i64) -> (i64)
; CHECK-NEXT:  unreachable
; CHECK-NEXT:  end_function

; CHECK-LABEL: .Lhas_i64_arg_bitcast_invalid.2:
; CHECK-NEXT:  .functype .Lhas_i64_arg_bitcast_invalid.2 (i32) -> (i32)
; CHECK-NEXT:  unreachable
; CHECK-NEXT:  end_function

; CHECK-LABEL: .Lhas_ptr_arg_bitcast_invalid:
; CHECK-NEXT: 	.functype .Lhas_ptr_arg_bitcast_invalid (i64) -> (i32)
; CHECK-NEXT: 	unreachable
; CHECK-NEXT: 	end_function

; CHECK-LABEL: .Lhas_ptr_arg_bitcast_invalid.4:
; CHECK-NEXT: 	.functype .Lhas_ptr_arg_bitcast_invalid.4 (i32) -> (i32)
; CHECK-NEXT: 	unreachable
; CHECK-NEXT: 	end_function
