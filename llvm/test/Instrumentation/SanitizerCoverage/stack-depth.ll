; This check verifies that stack depth instrumentation works correctly.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 \
; RUN:     -sanitizer-coverage-stack-depth -S | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 \
; RUN:     -sanitizer-coverage-stack-depth -sanitizer-coverage-trace-pc-guard \
; RUN:     -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__sancov_lowest_stack = thread_local(initialexec) global i64 -1
@__sancov_lowest_stack = thread_local global i64 0, align 8

define i32 @foo() {
entry:
; CHECK-LABEL: define i32 @foo
; CHECK-NOT: call i8* @llvm.frameaddress.p0i8(i32 0)
; CHECK-NOT: @__sancov_lowest_stack
; CHECK: ret i32 7

  ret i32 7
}

define i32 @bar() {
entry:
; CHECK-LABEL: define i32 @bar
; CHECK: [[framePtr:%[^ \t]+]] = call i8* @llvm.frameaddress.p0i8(i32 0)
; CHECK: [[frameInt:%[^ \t]+]] = ptrtoint i8* [[framePtr]] to [[intType:i[0-9]+]]
; CHECK: [[lowest:%[^ \t]+]] = load [[intType]], [[intType]]* @__sancov_lowest_stack
; CHECK: [[cmp:%[^ \t]+]] = icmp ult [[intType]] [[frameInt]], [[lowest]]
; CHECK: br i1 [[cmp]], label %[[ifLabel:[^ \t]+]], label
; CHECK: [[ifLabel]]:
; CHECK: store [[intType]] [[frameInt]], [[intType]]* @__sancov_lowest_stack
; CHECK: %call = call i32 @foo()
; CHECK: ret i32 %call

  %call = call i32 @foo()
  ret i32 %call
}

define weak_odr hidden i64* @_ZTW21__sancov_lowest_stack() {
  ret i64* @__sancov_lowest_stack
}
