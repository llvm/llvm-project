; RUN: opt -S -basicaa -dse < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

; We incorrectly returned noalias in the example below for "ptr.64" and
; "either_ptr.64".
; PR18460

%nested = type { %nested.i64 }
%nested.i64 = type { i64 }

define i64 @testcase(%nested * noalias %p1, %nested * noalias %p2,
                     i32 %a, i32 %b) {
  %ptr = getelementptr inbounds %nested* %p1, i64 -1, i32 0
  %ptr.64 = getelementptr inbounds %nested.i64* %ptr, i64 0, i32 0
  %ptr2= getelementptr inbounds %nested* %p2, i64 0, i32 0
  %cmp = icmp ult i32 %a, %b
  %either_ptr = select i1 %cmp, %nested.i64* %ptr2, %nested.i64* %ptr
  %either_ptr.64 = getelementptr inbounds %nested.i64* %either_ptr, i64 0, i32 0

; Because either_ptr.64 and ptr.64 can alias (we used to return noalias)
; elimination of the first store is not valid.

; CHECK: store i64 2
; CHECK: load
; CHECK; store i64 1

  store i64 2, i64* %ptr.64, align 8
  %r = load i64* %either_ptr.64, align 8
  store i64 1, i64* %ptr.64, align 8
  ret i64 %r
}
