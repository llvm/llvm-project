; RUN: opt -S -wasm-add-missing-prototypes %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: @foo_addr = global i64 (i32)* @foo, align 8
@foo_addr = global i64 (i32)* bitcast (i64 (...)* @foo to i64 (i32)*), align 8

; CHECK: @foo_addr_i8 = global i8* bitcast (i64 (i32)* @foo to i8*), align 8
@foo_addr_i8 = global i8* bitcast (i64 (...)* @foo to i8*), align 8

; CHECK-LABEL: @call_foo
; CHECK: %call = call i64 @foo(i32 42)
define void @call_foo(i32 %a) {
  %call = call i64 bitcast (i64 (...)* @foo to i64 (i32)*)(i32 42)
  ret void
}

; CHECK-LABEL: @call_foo_ptr
; CHECK: %1 = bitcast i64 (...)* bitcast (i64 (i32)* @foo to i64 (...)*) to i64 (i32)*
; CHECK-NEXT: %call = call i64 %1(i32 43)
define i64 @call_foo_ptr(i32 %a) {
  %1 = bitcast i64 (...)* @foo to i64 (i32)*
  %call = call i64 (i32) %1(i32 43)
  ret i64 %call
}

; CHECK-LABEL: @to_intptr_inst
; CHECK: %1 = bitcast i64 (...)* bitcast (i64 (i32)* @foo to i64 (...)*) to i8*
; CHECK-NEXT: ret i8* %1
define i8* @to_intptr_inst() {
  %1 = bitcast i64 (...)* @foo to i8*
  ret i8* %1
}

; CHECK-LABEL: @to_intptr_constexpr
; CHECK: ret i8* bitcast (i64 (i32)* @foo to i8*)
define i8* @to_intptr_constexpr() {
  ret i8* bitcast (i64 (...)* @foo to i8*)
}

; CHECK-LABEL: @null_compare
; CHECK: br i1 icmp eq (i64 (...)* bitcast (i64 (i32)* @foo to i64 (...)*), i64 (...)* null), label %if.then, label %if.end
define i8 @null_compare() {
  br i1 icmp eq (i64 (...)* @foo, i64 (...)* null), label %if.then, label %if.end
if.then:
  ret i8 0
if.end:
  ret i8 1
}

; CHECK-LABEL: @as_paramater
; CHECK: call void @func_param(i64 (...)* bitcast (i64 (i32)* @foo to i64 (...)*))
define void @as_paramater() {
  call void @func_param(i64 (...)* @foo)
  ret void
}

declare void @func_param(i64 (...)*)

; CHECK: declare extern_weak i64 @foo(i32)
declare extern_weak i64 @foo(...) #1

; CHECK-NOT: attributes {{.*}} = { {{.*}}"no-prototype"{{.*}} }
attributes #1 = { "no-prototype" }
