; To test that safestack does not break the musttail call contract.
;
; RUN: opt < %s --safe-stack -S | FileCheck %s
; RUN: opt < %s -passes=safe-stack -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare i32 @foo(ptr %p)
declare void @alloca_test_use(ptr)

define i32 @call_foo(ptr %a) safestack {
; CHECK-LABEL: @call_foo(
; CHECK-NEXT:    [[UNSAFE_STACK_PTR:%.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[UNSAFE_STACK_STATIC_TOP:%.*]] = getelementptr i8, ptr [[UNSAFE_STACK_PTR]], i32 -16
; CHECK-NEXT:    store ptr [[UNSAFE_STACK_STATIC_TOP]], ptr @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[UNSAFE_STACK_PTR]], i32 -10
; CHECK-NEXT:    call void @alloca_test_use(ptr [[TMP1]])
; CHECK-NEXT:    store ptr [[UNSAFE_STACK_PTR]], ptr @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[R:%.*]] = musttail call i32 @foo(ptr [[A:%.*]])
; CHECK-NEXT:    ret i32 [[R]]
;
  %x = alloca [10 x i8], align 1
  call void @alloca_test_use(ptr %x)
  %r = musttail call i32 @foo(ptr %a)
  ret i32 %r
}

define i32 @call_foo_cast(ptr %a) safestack {
; CHECK-LABEL: @call_foo_cast(
; CHECK-NEXT:    [[UNSAFE_STACK_PTR:%.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[UNSAFE_STACK_STATIC_TOP:%.*]] = getelementptr i8, ptr [[UNSAFE_STACK_PTR]], i32 -16
; CHECK-NEXT:    store ptr [[UNSAFE_STACK_STATIC_TOP]], ptr @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[UNSAFE_STACK_PTR]], i32 -10
; CHECK-NEXT:    call void @alloca_test_use(ptr [[TMP1]])
; CHECK-NEXT:    store ptr [[UNSAFE_STACK_PTR]], ptr @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[R:%.*]] = musttail call i32 @foo(ptr [[A:%.*]])
; CHECK-NEXT:    [[T:%.*]] = bitcast i32 [[R]] to i32
; CHECK-NEXT:    ret i32 [[T]]
;
  %x = alloca [10 x i8], align 1
  call void @alloca_test_use(ptr %x)
  %r = musttail call i32 @foo(ptr %a)
  %t = bitcast i32 %r to i32
  ret i32 %t
}
