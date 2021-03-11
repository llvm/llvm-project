; RUN: opt < %s -ipsccp -S | FileCheck %s
; Return value can't be zapped if there is a call that has operand bundle
; "clang.arc.attachedcall".

@g0 = global i8 zeroinitializer, align 1

; CHECK-LABEL: @foo(
; CHECK: ret i8* @g0

define internal i8* @foo() {
  ret i8* @g0
}

; CHECK-LABEL: @test(
; CHECK: %[[R:.*]] = call i8* @foo()
; CHECK call void (...) @llvm.objc.clang.arc.noop.use(i8* %[[R]])

define void @test() {
  %r = call i8* @foo() [ "clang.arc.attachedcall"(i64 1) ]
  call void (...) @llvm.objc.clang.arc.noop.use(i8* %r)
  ret void
}

declare void @llvm.objc.clang.arc.noop.use(...)
