; RUN: opt -S -wasm-add-missing-prototypes %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: @foo_addr = global ptr @foo, align 8
@foo_addr = global ptr @foo, align 8

; CHECK: @foo_addr_i8 = global ptr @foo, align 8
@foo_addr_i8 = global ptr @foo, align 8

; CHECK-LABEL: @call_foo
; CHECK: %call = call i64 @foo(i32 42)
define void @call_foo(i32 %a) {
  %call = call i64 @foo(i32 42)
  ret void
}

; CHECK-LABEL: @call_foo_ptr
; CHECK-NEXT: %call = call i64 @foo(i32 43)
define i64 @call_foo_ptr(i32 %a) {
  %call = call i64 (i32) @foo(i32 43)
  ret i64 %call
}

; CHECK-LABEL: @to_intptr_inst
; CHECK-NEXT: ret ptr @foo
define ptr @to_intptr_inst() {
  ret ptr @foo
}

; CHECK-LABEL: @to_intptr_constexpr
; CHECK: ret ptr @foo
define ptr @to_intptr_constexpr() {
  ret ptr @foo
}

; CHECK-LABEL: @null_compare
; CHECK: %cmp = icmp eq ptr @foo, null
; CHECK: br i1 %cmp, label %if.then, label %if.end
define i8 @null_compare() {
  %cmp = icmp eq ptr @foo, null
  br i1 %cmp, label %if.then, label %if.end
if.then:
  ret i8 0
if.end:
  ret i8 1
}

; CHECK-LABEL: @as_paramater
; CHECK: call void @func_param(ptr @foo)
define void @as_paramater() {
  call void @func_param(ptr @foo)
  ret void
}

; Check if a sret parameter works in a no-prototype function.
; CHECK-LABEL: @sret_param
; CHECK: call void @make_struct_foo(ptr sret(%struct.foo) %foo)
%struct.foo = type { i32, i32 }
declare void @make_struct_foo(ptr sret(%struct.foo), ...) #1
define void @sret_param() {
  %foo = alloca %struct.foo, align 4
  call void @make_struct_foo(ptr sret(%struct.foo) %foo)
  ret void
}

declare void @func_param(ptr)

; CHECK: declare void @func_not_called()
declare void @func_not_called(...) #1

; CHECK: declare extern_weak i64 @foo(i32)
declare extern_weak i64 @foo(...) #1

; CHECK-NOT: attributes {{.*}} = { {{.*}}"no-prototype"{{.*}} }
attributes #1 = { "no-prototype" }
