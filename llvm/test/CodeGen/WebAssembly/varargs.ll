; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test varargs constructs.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Test va_start.

; TODO: Test va_start.

;define void @start(i8** %ap, ...) {
;entry:
;  %0 = bitcast i8** %ap to i8*
;  call void @llvm.va_start(i8* %0)
;  ret void
;}

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
; CHECK-NEXT: i32.store $discard=, 0($0), $pop0{{$}}
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
; CHECK-NEXT: .local     i32{{$}}
; CHECK-NEXT: i32.load   $1=, 0($0){{$}}
; CHECK-NEXT: i32.const  $push0=, 4{{$}}
; CHECK-NEXT: i32.add    $push1=, $1, $pop0{{$}}
; CHECK-NEXT: i32.store  $discard=, 0($0), $pop1{{$}}
; CHECK-NEXT: i32.load   $push2=, 0($1){{$}}
; CHECK-NEXT: return     $pop2{{$}}
define i8 @arg_i8(i8** %ap) {
entry:
  %t = va_arg i8** %ap, i8
  ret i8 %t
}

; Test va_arg with an i32 argument.

; CHECK-LABEL: arg_i32:
; CHECK-NEXT: .param     i32{{$}}
; CHECK-NEXT: .result    i32{{$}}
; CHECK-NEXT: .local     i32{{$}}
; CHECK-NEXT: i32.load   $push0=, 0($0){{$}}
; CHECK-NEXT: i32.const  $push1=, 3{{$}}
; CHECK-NEXT: i32.add    $push2=, $pop0, $pop1{{$}}
; CHECK-NEXT: i32.const  $push3=, -4{{$}}
; CHECK-NEXT: i32.and    $1=, $pop2, $pop3{{$}}
; CHECK-NEXT: i32.const  $push4=, 4{{$}}
; CHECK-NEXT: i32.add    $push5=, $1, $pop4{{$}}
; CHECK-NEXT: i32.store  $discard=, 0($0), $pop5{{$}}
; CHECK-NEXT: i32.load   $push6=, 0($1){{$}}
; CHECK-NEXT: return     $pop6{{$}}
define i32 @arg_i32(i8** %ap) {
entry:
  %t = va_arg i8** %ap, i32
  ret i32 %t
}

; Test va_arg with an i128 argument.

; CHECK-LABEL: arg_i128:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: .local
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
; CHECK-NEXT: call callee{{$}}
; CHECK-NEXT: return{{$}}
define void @caller_none() {
  call void (...) @callee()
  ret void
}

; TODO: Test a varargs call with actual arguments.

;define void @caller_some() {
;  call void (...) @callee(i32 0, double 2.0)
;  ret void
;}

declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)
declare void @llvm.va_copy(i8*, i8*)
