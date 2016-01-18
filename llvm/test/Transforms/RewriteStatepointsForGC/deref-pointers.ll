; RUN: opt -S -rewrite-statepoints-for-gc < %s | FileCheck %s

declare void @foo()
declare i8 addrspace(1)* @some_function()
declare void @some_function_consumer(i8 addrspace(1)*)
declare dereferenceable(4) i8 addrspace(1)* @some_function_ret_deref()
; CHECK: declare i8 addrspace(1)* @some_function_ret_deref()
declare noalias i8 addrspace(1)* @some_function_ret_noalias()
; CHECK: declare i8 addrspace(1)* @some_function_ret_noalias()

define i8 addrspace(1)* @test_deref_arg(i8 addrspace(1)* dereferenceable(4) %a) gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @test_deref_arg(i8 addrspace(1)* %a)
entry:
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_deref_or_null_arg(i8 addrspace(1)* dereferenceable_or_null(4) %a) gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @test_deref_or_null_arg(i8 addrspace(1)* %a)
entry:
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_deref_retval() gc "statepoint-example" {
; CHECK-LABEL: @test_deref_retval(
entry:
  %a = call dereferenceable(4) i8 addrspace(1)* @some_function()
; CHECK: %a = call i8 addrspace(1)* @some_function()
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_deref_or_null_retval() gc "statepoint-example" {
; CHECK-LABEL: @test_deref_or_null_retval(
entry:
  %a = call dereferenceable_or_null(4) i8 addrspace(1)* @some_function()
; CHECK: %a = call i8 addrspace(1)* @some_function()
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %a
}

define i8 @test_md(i8 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_md(
 entry:
; CHECK: %tmp = load i8, i8 addrspace(1)* %ptr, !tbaa !0
  %tmp = load i8, i8 addrspace(1)* %ptr, !tbaa !0
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 %tmp
}

define i8 addrspace(1)* @test_decl_only_attribute(i8 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_decl_only_attribute(
entry:
; No change here, but the prototype of some_function_ret_deref should have changed.
; CHECK: call i8 addrspace(1)* @some_function_ret_deref()
  %a = call i8 addrspace(1)* @some_function_ret_deref()
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_callsite_arg_attribute(i8 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_callsite_arg_attribute(
entry:
; CHECK: call void @some_function_consumer(i8 addrspace(1)* %ptr)
  call void @some_function_consumer(i8 addrspace(1)* dereferenceable(4) %ptr)
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %ptr
}

define i8 addrspace(1)* @test_noalias_arg(i8 addrspace(1)* noalias %a) gc "statepoint-example" {
; CHECK: define i8 addrspace(1)* @test_noalias_arg(i8 addrspace(1)* %a)
entry:
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_noalias_retval() gc "statepoint-example" {
; CHECK-LABEL: @test_noalias_retval(
entry:
  %a = call noalias i8 addrspace(1)* @some_function()
; CHECK: %a = call i8 addrspace(1)* @some_function()
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_decl_only_noalias(i8 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_decl_only_noalias(
entry:
; No change here, but the prototype of some_function_ret_noalias should have changed.
; CHECK: call i8 addrspace(1)* @some_function_ret_noalias()
  %a = call i8 addrspace(1)* @some_function_ret_noalias()
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_callsite_arg_noalias(i8 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_callsite_arg_noalias(
entry:
; CHECK: call void @some_function_consumer(i8 addrspace(1)* %ptr)
  call void @some_function_consumer(i8 addrspace(1)* noalias %ptr)
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %ptr
}

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

!0 = !{!1, !1, i64 0, i64 1}
!1 = !{!"red", !2}
!2 = !{!"blue"}

; CHECK: !0 = !{!1, !1, i64 0}
; CHECK: !1 = !{!"red", !2}
; CHECK: !2 = !{!"blue"}
