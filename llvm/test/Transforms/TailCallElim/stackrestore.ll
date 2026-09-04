; RUN: opt -S -passes=tailcallelim < %s | FileCheck %s

define void @foo() {
; CHECK-LABEL: define void @foo()
; CHECK-NOT:   tail call void @llvm.stackrestore.p0
;
entry:
  %0 = call ptr @llvm.stacksave.p0()
  call void @llvm.stackrestore.p0(ptr %0)
  ret void
}

; llvm.stackrestore does not capture its argument, so it does not make the
; local stack escape and the call after it is still a tail call. The token
; reaches it directly here and through a phi below, which is the shape clang
; emits for a variable length array declared in both arms of an if.
define void @stackrestore_is_not_an_escape(ptr %p) {
; CHECK-LABEL: define void @stackrestore_is_not_an_escape(
; CHECK:         tail call void @callee
;
entry:
  %ss = call ptr @llvm.stacksave.p0()
  call void @llvm.stackrestore.p0(ptr %ss)
  call void @callee(ptr %p)
  ret void
}

define void @stackrestore_of_a_phi_is_not_an_escape(ptr %p, i1 %c) {
; CHECK-LABEL: define void @stackrestore_of_a_phi_is_not_an_escape(
; CHECK:         tail call void @callee
;
entry:
  %ss = call ptr @llvm.stacksave.p0()
  br i1 %c, label %then, label %else

then:
  %ss2 = call ptr @llvm.stacksave.p0()
  br label %join

else:
  br label %join

join:
  %sink = phi ptr [ %ss2, %then ], [ %ss, %else ]
  call void @llvm.stackrestore.p0(ptr %sink)
  call void @callee(ptr %p)
  ret void
}

; An alloca handed to llvm.stackrestore is not captured by it either.
define void @stackrestore_of_an_alloca_is_not_an_escape(ptr %p) {
; CHECK-LABEL: define void @stackrestore_of_an_alloca_is_not_an_escape(
; CHECK:         tail call void @callee
;
entry:
  %a = alloca i8
  call void @llvm.stackrestore.p0(ptr %a)
  call void @callee(ptr %p)
  ret void
}

declare void @callee(ptr)
declare ptr @llvm.stacksave.p0()
declare void @llvm.stackrestore.p0(ptr)
