; RUN: opt -safe-stack -S -mtriple=x86_64-unknown < %s -o - | FileCheck %s
; RUN: opt -passes=safe-stack -S -mtriple=x86_64-unknown < %s -o - | FileCheck %s

define void @foo() safestack sspreq {
entry:
; CHECK: %[[USP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
; CHECK: %[[USST:.*]] = getelementptr i8, ptr %[[USP]], i32 -16
; CHECK: store ptr %[[USST]], ptr @__safestack_unsafe_stack_ptr

; CHECK: %[[A:.*]] = getelementptr i8, ptr %[[USP]], i32 -8
; CHECK: %[[StackGuard:.*]] = call ptr @llvm.stackguard()
; CHECK: store ptr %[[StackGuard]], ptr %[[A]]
  %a = alloca i8, align 1

; CHECK: call void @Capture
  call void @Capture(ptr %a)

; CHECK: %[[B:.*]] = load ptr, ptr %[[A]]
; CHECK: %[[COND:.*]] = icmp ne ptr %[[StackGuard]], %[[B]]
; CHECK: br i1 %[[COND]], {{.*}} !prof

; CHECK:      call void @__stack_chk_fail()
; CHECK-NEXT: unreachable

; CHECK:      store ptr %[[USP]], ptr @__safestack_unsafe_stack_ptr
; CHECK-NEXT: ret void
  ret void
}

declare void @Capture(ptr)
