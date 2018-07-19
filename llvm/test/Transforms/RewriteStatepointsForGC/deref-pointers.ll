; RUN: opt -S -rewrite-statepoints-for-gc < %s | FileCheck %s
; RUN: opt -S -passes=rewrite-statepoints-for-gc < %s | FileCheck %s

; CHECK: declare i8 addrspace(1)* @some_function_ret_deref()
; CHECK: define i8 addrspace(1)* @test_deref_arg(i8 addrspace(1)* %a)
; CHECK: define i8 addrspace(1)* @test_deref_or_null_arg(i8 addrspace(1)* %a)
; CHECK: define i8 addrspace(1)* @test_noalias_arg(i8 addrspace(1)* %a)

declare void @foo()

declare i8 addrspace(1)* @some_function() "gc-leaf-function"

declare void @some_function_consumer(i8 addrspace(1)*) "gc-leaf-function"

declare dereferenceable(4) i8 addrspace(1)* @some_function_ret_deref() "gc-leaf-function"
declare noalias i8 addrspace(1)* @some_function_ret_noalias() "gc-leaf-function"

define i8 addrspace(1)* @test_deref_arg(i8 addrspace(1)* dereferenceable(4) %a) gc "statepoint-example" {
entry:
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_deref_or_null_arg(i8 addrspace(1)* dereferenceable_or_null(4) %a) gc "statepoint-example" {
entry:
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_noalias_arg(i8 addrspace(1)* noalias %a) gc "statepoint-example" {
entry:
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_deref_retval() gc "statepoint-example" {
; CHECK-LABEL: @test_deref_retval(
; CHECK: %a = call i8 addrspace(1)* @some_function()
entry:
  %a = call dereferenceable(4) i8 addrspace(1)* @some_function()
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_deref_or_null_retval() gc "statepoint-example" {
; CHECK-LABEL: @test_deref_or_null_retval(
; CHECK: %a = call i8 addrspace(1)* @some_function()
entry:
  %a = call dereferenceable_or_null(4) i8 addrspace(1)* @some_function()
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_noalias_retval() gc "statepoint-example" {
; CHECK-LABEL: @test_noalias_retval(
; CHECK: %a = call i8 addrspace(1)* @some_function()
entry:
  %a = call noalias i8 addrspace(1)* @some_function()
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 addrspace(1)* %a
}

define i8 @test_md(i8 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_md(
; CHECK: %tmp = load i8, i8 addrspace(1)* %ptr, !tbaa !0
entry:
  %tmp = load i8, i8 addrspace(1)* %ptr, !tbaa !0
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 %tmp
}

define i8 addrspace(1)* @test_decl_only_attribute(i8 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_decl_only_attribute(
; No change here, but the prototype of some_function_ret_deref should have changed.
; CHECK: call i8 addrspace(1)* @some_function_ret_deref()
entry:
  %a = call i8 addrspace(1)* @some_function_ret_deref()
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_decl_only_noalias(i8 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_decl_only_noalias(
; No change here, but the prototype of some_function_ret_noalias should have changed.
; CHECK: call i8 addrspace(1)* @some_function_ret_noalias()
entry:
  %a = call i8 addrspace(1)* @some_function_ret_noalias()
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 addrspace(1)* %a
}

define i8 addrspace(1)* @test_callsite_arg_attribute(i8 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_callsite_arg_attribute(
; CHECK: call void @some_function_consumer(i8 addrspace(1)* %ptr)
; CHECK: !0 = !{!1, !1, i64 0}
; CHECK: !1 = !{!"red", !2}
; CHECK: !2 = !{!"blue"}
entry:
  call void @some_function_consumer(i8 addrspace(1)* dereferenceable(4) noalias %ptr)
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 addrspace(1)* %ptr
}
!0 = !{!1, !1, i64 0, i64 1}
!1 = !{!"red", !2}
!2 = !{!"blue"}
