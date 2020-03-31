; RUN: opt < %s -inline-threshold=0 -always-inline -S | FileCheck %s
; RUN: opt < %s -passes=always-inline -S | FileCheck %s

declare i8* @foo(i8*) argmemonly nounwind

define i8* @callee(i8 *%p) alwaysinline {
; CHECK: @callee(
; CHECK: call i8* @foo(i8* noalias %p)
  %r = call i8* @foo(i8* noalias %p)
  ret i8* %r
}

define i8* @caller(i8* %ptr, i64 %x) {
; CHECK-LABEL: @caller
; CHECK: call nonnull i8* @foo(i8* noalias
  %gep = getelementptr inbounds i8, i8* %ptr, i64 %x
  %p = call nonnull i8* @callee(i8* %gep)
  ret i8* %p
}

declare void @llvm.experimental.guard(i1,...)
; Cannot add nonnull attribute to foo
; because the guard is a throwing call
define internal i8* @callee_with_throwable(i8* %p) alwaysinline {
; CHECK-NOT: callee_with_throwable
  %r = call i8* @foo(i8* %p)
  %cond = icmp ne i8* %r, null
  call void (i1, ...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
  ret i8* %r
}

declare i8* @bar(i8*) readonly nounwind
; Here also we cannot add nonnull attribute to the call bar.
define internal i8* @callee_with_explicit_control_flow(i8* %p) alwaysinline {
; CHECK-NOT: callee_with_explicit_control_flow
  %r = call i8* @bar(i8* %p)
  %cond = icmp ne i8* %r, null
  br i1 %cond, label %ret, label %orig

ret:
  ret i8* %r

orig:
  ret i8* %p
}

define i8* @caller2(i8* %ptr, i64 %x, i1 %cond) {
; CHECK-LABEL: @caller2
; CHECK: call i8* @foo
; CHECK: call i8* @bar
  %gep = getelementptr inbounds i8, i8* %ptr, i64 %x
  %p = call nonnull i8* @callee_with_throwable(i8* %gep)
  %q = call nonnull i8* @callee_with_explicit_control_flow(i8* %gep)
  br i1 %cond, label %pret, label %qret

pret:
  ret i8* %p

qret:
  ret i8* %q
}

define internal i8* @callee3(i8 *%p) alwaysinline {
; CHECK-NOT: callee3
  %r = call noalias i8* @foo(i8* %p)
  ret i8* %r
}

; add the deref attribute to the existing attributes on foo.
define i8* @caller3(i8* %ptr, i64 %x) {
; CHECK-LABEL: caller3
; CHECK: call noalias dereferenceable_or_null(12) i8* @foo
  %gep = getelementptr inbounds i8, i8* %ptr, i64 %x
  %p = call dereferenceable_or_null(12) i8* @callee3(i8* %gep)
  ret i8* %p
}

declare i8* @inf_loop_call(i8*) nounwind
; We cannot propagate attributes to foo because we do not know whether inf_loop_call
; will return execution.
define internal i8* @callee_with_sideeffect_callsite(i8* %p) alwaysinline {
; CHECK-NOT: callee_with_sideeffect_callsite
  %r = call i8* @foo(i8* %p)
  %v = call i8* @inf_loop_call(i8* %p)
  ret i8* %r
}

; do not add deref attribute to foo
define i8* @test4(i8* %ptr, i64 %x) {
; CHECK-LABEL: test4
; CHECK: call i8* @foo
  %gep = getelementptr inbounds i8, i8* %ptr, i64 %x
  %p = call dereferenceable_or_null(12) i8* @callee_with_sideeffect_callsite(i8* %gep)
  ret i8* %p
}

declare i8* @baz(i8*) nounwind readonly
define internal i8* @callee5(i8* %p) alwaysinline {
; CHECK-NOT: callee5
  %r = call i8* @foo(i8* %p)
  %v = call i8* @baz(i8* %p)
  ret i8* %r
}

; add the deref attribute to foo.
define i8* @test5(i8* %ptr, i64 %x) {
; CHECK-LABEL: test5
; CHECK: call dereferenceable_or_null(12) i8* @foo
  %gep = getelementptr inbounds i8, i8* %ptr, i64 %x
  %s = call dereferenceable_or_null(12) i8* @callee5(i8* %gep)
  ret i8* %s
}

; deref attributes have different values on the callee and the call feeding into
; the return.
; AttrBuilder chooses the already existing value and does not overwrite it.
define internal i8* @callee6(i8* %p) alwaysinline {
; CHECK-NOT: callee6
  %r = call dereferenceable_or_null(16) i8* @foo(i8* %p)
  %v = call i8* @baz(i8* %p)
  ret i8* %r
}


define i8* @test6(i8* %ptr, i64 %x) {
; CHECK-LABEL: test6
; CHECK: call dereferenceable_or_null(16) i8* @foo
  %gep = getelementptr inbounds i8, i8* %ptr, i64 %x
  %s = call dereferenceable_or_null(12) i8* @callee6(i8* %gep)
  ret i8* %s
}

; We add the attributes from the callee to both the calls below.
define internal i8* @callee7(i8 *%ptr, i1 %cond) alwaysinline {
; CHECK-NOT: @callee7(
  br i1 %cond, label %pass, label %fail

pass:
  %r = call i8* @foo(i8* noalias %ptr)
  ret i8* %r

fail:
  %s = call i8* @baz(i8* %ptr)
  ret i8* %s
}

define void @test7(i8* %ptr, i64 %x, i1 %cond) {
; CHECK-LABEL: @test7
; CHECK: call nonnull i8* @foo(i8* noalias
; CHECK: call nonnull i8* @baz
; CHECK: phi i8*
; CHECK: call void @snort

  %gep = getelementptr inbounds i8, i8* %ptr, i64 %x
  %t = call nonnull i8* @callee7(i8* %gep, i1 %cond)
  call void @snort(i8* %t)
  ret void
}
declare void @snort(i8*)
