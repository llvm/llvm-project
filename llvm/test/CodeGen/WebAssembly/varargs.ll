; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -verify-machineinstrs | FileCheck %s

; Test varargs constructs.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Test va_start.

; TODO: Test va_start.
; CHECK-LABEL: start:
; CHECK-NEXT: .param i32, i32
; CHECK-NOT: __stack_pointer
define void @start(i8** %ap, ...) {
entry:
  %0 = bitcast i8** %ap to i8*
; Store the second argument (the hidden vararg buffer pointer) into ap
; CHECK: i32.store 0($0), $1
  call void @llvm.va_start(i8* %0)
  ret void
}

; Test va_end.

; CHECK-LABEL: end:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: return{{$}}
define void @end(i8** %ap) {
entry:
  %0 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %0)
  ret void
}

; Test va_copy.

; CHECK-LABEL: copy:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.load  $push0=, 0($1){{$}}
; CHECK-NEXT: i32.store 0($0), $pop0{{$}}
; CHECK-NEXT: return{{$}}
define void @copy(i8** %ap, i8** %bp) {
entry:
  %0 = bitcast i8** %ap to i8*
  %1 = bitcast i8** %bp to i8*
  call void @llvm.va_copy(i8* %0, i8* %1)
  ret void
}

; Test va_arg with an i8 argument.

; CHECK-LABEL: arg_i8:
; CHECK-NEXT: .param     i32{{$}}
; CHECK-NEXT: .result    i32{{$}}
; CHECK-NEXT: i32.load   $push[[NUM0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: tee_local  $push[[NUM1:[0-9]+]]=, $1=, $pop[[NUM0]]{{$}}
; CHECK-NEXT: i32.const  $push[[NUM2:[0-9]+]]=, 4{{$}}
; CHECK-NEXT: i32.add    $push[[NUM3:[0-9]+]]=, $pop[[NUM1]], $pop[[NUM2]]{{$}}
; CHECK-NEXT: i32.store  0($0), $pop[[NUM3]]{{$}}
; CHECK-NEXT: i32.load   $push[[NUM4:[0-9]+]]=, 0($1){{$}}
; CHECK-NEXT: return     $pop[[NUM4]]{{$}}
define i8 @arg_i8(i8** %ap) {
entry:
  %t = va_arg i8** %ap, i8
  ret i8 %t
}

; Test va_arg with an i32 argument.

; CHECK-LABEL: arg_i32:
; CHECK-NEXT: .param     i32{{$}}
; CHECK-NEXT: .result    i32{{$}}
; CHECK-NEXT: i32.load   $push[[NUM0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: i32.const  $push[[NUM1:[0-9]+]]=, 3{{$}}
; CHECK-NEXT: i32.add    $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: i32.const  $push[[NUM3:[0-9]+]]=, -4{{$}}
; CHECK-NEXT: i32.and    $push[[NUM4:[0-9]+]]=, $pop[[NUM2]], $pop[[NUM3]]{{$}}
; CHECK-NEXT: tee_local  $push[[NUM5:[0-9]+]]=, $1=, $pop[[NUM4]]{{$}}
; CHECK-NEXT: i32.const  $push[[NUM6:[0-9]+]]=, 4{{$}}
; CHECK-NEXT: i32.add    $push[[NUM7:[0-9]+]]=, $pop[[NUM5]], $pop[[NUM6]]{{$}}
; CHECK-NEXT: i32.store  0($0), $pop[[NUM7]]{{$}}
; CHECK-NEXT: i32.load   $push[[NUM8:[0-9]+]]=, 0($1){{$}}
; CHECK-NEXT: return     $pop[[NUM8]]{{$}}
define i32 @arg_i32(i8** %ap) {
entry:
  %t = va_arg i8** %ap, i32
  ret i32 %t
}

; Test va_arg with an i128 argument.

; CHECK-LABEL: arg_i128:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK: i32.and
; CHECK: i64.load
; CHECK: i64.load
; CHECK: return{{$}}
define i128 @arg_i128(i8** %ap) {
entry:
  %t = va_arg i8** %ap, i128
  ret i128 %t
}

; Test a varargs call with no actual arguments.

declare void @callee(...)

; CHECK-LABEL: caller_none:
; CHECK-NEXT: i32.const $push0=, 0
; CHECK-NEXT: call callee@FUNCTION, $pop0
; CHECK-NEXT: return{{$}}
define void @caller_none() {
  call void (...) @callee()
  ret void
}

; Test a varargs call with some actual arguments.
; Note that the store of 2.0 is converted to an i64 store; this optimization
; is not needed on WebAssembly, but there isn't currently a convenient hook for
; disabling it.

; CHECK-LABEL: caller_some
; CHECK-DAG: i32.store
; CHECK-DAG: i64.store
define void @caller_some() {
  call void (...) @callee(i32 0, double 2.0)
  ret void
}

; Test a va_start call in a non-entry block
; CHECK-LABEL: startbb:
; CHECK: .param i32, i32, i32
define void @startbb(i1 %cond, i8** %ap, ...) {
entry:
  br i1 %cond, label %bb0, label %bb1
bb0:
  ret void
bb1:
  %0 = bitcast i8** %ap to i8*
; Store the second argument (the hidden vararg buffer pointer) into ap
; CHECK: i32.store 0($1), $2
  call void @llvm.va_start(i8* %0)
  ret void
}


declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)
declare void @llvm.va_copy(i8*, i8*)
