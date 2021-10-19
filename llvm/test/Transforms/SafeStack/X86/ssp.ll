; RUN: opt -safe-stack -S -mtriple=x86_64-unknown < %s -o - | FileCheck %s

define void @foo() safestack sspreq {
entry:
; CHECK: %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK: %[[USST:.*]] = getelementptr i8, i8* %[[USP]], i32 -16
; CHECK: store i8* %[[USST]], i8** @__safestack_unsafe_stack_ptr

; CHECK: %[[A:.*]] = getelementptr i8, i8* %[[USP]], i32 -8
; CHECK: %[[StackGuardSlot:.*]] = bitcast i8* %[[A]] to i8**
; CHECK: %[[StackGuard:.*]] = call i8* @llvm.stackguard()
; CHECK: store i8* %[[StackGuard]], i8** %[[StackGuardSlot]]
  %a = alloca i8, align 1

; CHECK: call void @Capture
  call void @Capture(i8* %a)

; CHECK: %[[B:.*]] = load i8*, i8** %[[StackGuardSlot]]
; CHECK: %[[COND:.*]] = icmp ne i8* %[[StackGuard]], %[[B]]
; CHECK: br i1 %[[COND]], {{.*}} !prof

; CHECK:      call void @__stack_chk_fail()
; CHECK-NEXT: unreachable

; CHECK:      store i8* %[[USP]], i8** @__safestack_unsafe_stack_ptr
; CHECK-NEXT: ret void
  ret void
}

declare void @Capture(i8*)
