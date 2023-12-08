; RUN: opt -S -passes=rewrite-statepoints-for-gc < %s | FileCheck %s

; CHECK: declare ptr addrspace(1) @some_function_ret_deref()
; CHECK: define ptr addrspace(1) @test_deref_arg(ptr addrspace(1) %a)
; CHECK: define ptr addrspace(1) @test_deref_or_null_arg(ptr addrspace(1) %a)
; CHECK: define ptr addrspace(1) @test_noalias_arg(ptr addrspace(1) %a)

declare void @foo()

declare ptr addrspace(1) @some_function() "gc-leaf-function"

declare void @some_function_consumer(ptr addrspace(1)) "gc-leaf-function"

declare dereferenceable(4) ptr addrspace(1) @some_function_ret_deref() "gc-leaf-function"
declare noalias ptr addrspace(1) @some_function_ret_noalias() "gc-leaf-function"

define ptr addrspace(1) @test_deref_arg(ptr addrspace(1) dereferenceable(4) %a) gc "statepoint-example" {
entry:
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret ptr addrspace(1) %a
}

define ptr addrspace(1) @test_deref_or_null_arg(ptr addrspace(1) dereferenceable_or_null(4) %a) gc "statepoint-example" {
entry:
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret ptr addrspace(1) %a
}

define ptr addrspace(1) @test_noalias_arg(ptr addrspace(1) noalias %a) gc "statepoint-example" {
entry:
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret ptr addrspace(1) %a
}

define ptr addrspace(1) @test_deref_retval() gc "statepoint-example" {
; CHECK-LABEL: @test_deref_retval(
; CHECK: %a = call ptr addrspace(1) @some_function()
entry:
  %a = call dereferenceable(4) ptr addrspace(1) @some_function()
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret ptr addrspace(1) %a
}

define ptr addrspace(1) @test_deref_or_null_retval() gc "statepoint-example" {
; CHECK-LABEL: @test_deref_or_null_retval(
; CHECK: %a = call ptr addrspace(1) @some_function()
entry:
  %a = call dereferenceable_or_null(4) ptr addrspace(1) @some_function()
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret ptr addrspace(1) %a
}

define ptr addrspace(1) @test_noalias_retval() gc "statepoint-example" {
; CHECK-LABEL: @test_noalias_retval(
; CHECK: %a = call ptr addrspace(1) @some_function()
entry:
  %a = call noalias ptr addrspace(1) @some_function()
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret ptr addrspace(1) %a
}

define i8 @test_md(ptr addrspace(1) %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_md(
; CHECK: %tmp = load i8, ptr addrspace(1) %ptr, align 1, !tbaa [[TAG_old:!.*]]
entry:
  %tmp = load i8, ptr addrspace(1) %ptr, !tbaa !0
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 %tmp
}

; Same as test_md() above, but with new-format TBAA metadata.
define i8 @test_md_new(ptr addrspace(1) %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_md_new(
; CHECK: %tmp = load i8, ptr addrspace(1) %ptr, align 1, !tbaa [[TAG_new:!.*]]
entry:
  %tmp = load i8, ptr addrspace(1) %ptr, !tbaa !4
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 %tmp
}

define ptr addrspace(1) @test_decl_only_attribute(ptr addrspace(1) %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_decl_only_attribute(
; No change here, but the prototype of some_function_ret_deref should have changed.
; CHECK: call ptr addrspace(1) @some_function_ret_deref()
entry:
  %a = call ptr addrspace(1) @some_function_ret_deref()
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret ptr addrspace(1) %a
}

define ptr addrspace(1) @test_decl_only_noalias(ptr addrspace(1) %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_decl_only_noalias(
; No change here, but the prototype of some_function_ret_noalias should have changed.
; CHECK: call ptr addrspace(1) @some_function_ret_noalias()
entry:
  %a = call ptr addrspace(1) @some_function_ret_noalias()
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret ptr addrspace(1) %a
}

define ptr addrspace(1) @test_callsite_arg_attribute(ptr addrspace(1) %ptr) gc "statepoint-example" {
; CHECK-LABEL: @test_callsite_arg_attribute(
; CHECK: call void @some_function_consumer(ptr addrspace(1) %ptr)
entry:
  call void @some_function_consumer(ptr addrspace(1) dereferenceable(4) noalias %ptr)
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret ptr addrspace(1) %ptr
}

!0 = !{!1, !2, i64 0, i64 1}  ; TAG_old
!1 = !{!"type_base_old", !2, i64 0}
!2 = !{!"type_access_old", !3}
!3 = !{!"root"}

!4 = !{!5, !6, i64 0, i64 1, i64 1}  ; TAG_new
!5 = !{!3, i64 1, !"type_base_new", !6, i64 0, i64 1}
!6 = !{!3, i64 1, !"type_access_new"}

; CHECK-DAG: [[ROOT:!.*]] = !{!"root"}
; CHECK-DAG: [[TYPE_access_old:!.*]] = !{!"type_access_old", [[ROOT]]}
; CHECK-DAG: [[TYPE_base_old:!.*]] = !{!"type_base_old", [[TYPE_access_old]], i64 0}
; CHECK-DAG: [[TAG_old]] = !{[[TYPE_base_old]], [[TYPE_access_old]], i64 0}
; CHECK-DAG: [[TYPE_access_new:!.*]] = !{[[ROOT]], i64 1, !"type_access_new"}
; CHECK-DAG: [[TYPE_base_new:!.*]] = !{[[ROOT]], i64 1, !"type_base_new", [[TYPE_access_new]], i64 0, i64 1}
; CHECK-DAG: [[TAG_new]] = !{[[TYPE_base_new]], [[TYPE_access_new]], i64 0, i64 1}
