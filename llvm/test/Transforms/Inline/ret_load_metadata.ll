; RUN: opt < %s -inline-threshold=0 -update-load-metadata-during-inlining=true -always-inline -S | FileCheck %s
; RUN: opt < %s -passes=always-inline -update-load-metadata-during-inlining=true -S | FileCheck %s


define i8* @callee(i8** %p) alwaysinline {
; CHECK: @callee(
; CHECK-NOT: nonnull
  %r = load i8*, i8** %p, align 8
  ret i8* %r
}

define i8* @test(i8** %ptr, i64 %x) {
; CHECK-LABEL: @test
; CHECK: load i8*, i8** %gep, align 8, !nonnull !0
  %gep = getelementptr inbounds i8*, i8** %ptr, i64 %x
  %p = call nonnull i8* @callee(i8** %gep)
  ret i8* %p
}

declare void @does_not_return(i8*) nounwind
define internal i8* @callee_negative(i8** %p) alwaysinline {
; CHECK-NOT: @callee_negative(
  %r = load i8*, i8** %p, align 8
  call void @does_not_return(i8* %r)
  ret i8* %r
}

define i8* @negative_test(i8** %ptr, i64 %x) {
; CHECK-LABEL: @negative_test
; CHECK: load i8*, i8** %gep, align 8
; CHECK-NOT: nonnull
  %gep = getelementptr inbounds i8*, i8** %ptr, i64 %x
  %p = call nonnull i8* @callee_negative(i8** %gep)
  ret i8* %p
}


define internal i8* @callee2(i8** %p) alwaysinline {
; CHECK-NOT: @callee2(
  %r = load i8*, i8** %p, align 8
  ret i8* %r
}

; dereferenceable attribute in default addrspace implies nonnull
define i8* @test2(i8** %ptr, i64 %x) {
; CHECK-LABEL: @test2
; CHECK: load i8*, i8** %gep, align 8, !nonnull !0, !dereferenceable !1
  %gep = getelementptr inbounds i8*, i8** %ptr, i64 %x
  %p = call dereferenceable(12) i8* @callee(i8** %gep)
  ret i8* %p
}

declare void @bar(i8 addrspace(1)*) argmemonly nounwind

define internal i8 addrspace(1)* @callee3(i8 addrspace(1)* addrspace(1)* %p) alwaysinline {
  %r = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %p, align 8
  call void @bar(i8 addrspace(1)* %r)
  ret i8 addrspace(1)* %r
}

; This load in callee already has a dereferenceable_or_null metadata. We should
; honour it.
define internal i8 addrspace(1)* @callee5(i8 addrspace(1)* addrspace(1)* %p) alwaysinline {
  %r = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %p, align 8, !dereferenceable_or_null !3
  call void @bar(i8 addrspace(1)* %r)
  ret i8 addrspace(1)* %r
}

define i8 addrspace(1)* @test3(i8 addrspace(1)* addrspace(1)* %ptr, i64 %x) {
; CHECK-LABEL: @test3
; CHECK: load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %gep, align 8, !dereferenceable_or_null !2
; CHECK: load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %ptr, align 8, !dereferenceable_or_null !3
  %gep = getelementptr inbounds i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %ptr, i64 %x
  %p = call dereferenceable_or_null(16) i8 addrspace(1)* @callee3(i8 addrspace(1)* addrspace(1)* %gep)
  %q = call dereferenceable_or_null(20) i8 addrspace(1)* @callee5(i8 addrspace(1)* addrspace(1)* %ptr)
  ret i8 addrspace(1)* %p
}

; attribute is part of the callee itself
define nonnull i8* @callee4(i8** %p) alwaysinline {
  %r = load i8*, i8** %p, align 8
  ret i8* %r
}

; TODO : We should infer the attribute on the callee
; and add the nonnull on the load
define i8* @test4(i8** %ptr, i64 %x) {
; CHECK-LABEL: @test4
; CHECK: load i8*, i8** %gep, align 8
; CHECK-NOT: nonnull
  %gep = getelementptr inbounds i8*, i8** %ptr, i64 %x
  %p = call i8* @callee(i8** %gep)
  ret i8* %p
}

!0 = !{i64 1}
!1 = !{i64 12}
!2 = !{i64 16}
!3 = !{i64 24}
; CHECK: !0 = !{i64 1}
; CHECK: !1 = !{i64 12}
; CHECK: !2 = !{i64 16}
; CHECK: !3 = !{i64 24}
