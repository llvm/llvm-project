; To test that safestack does not break the musttail call contract.
;
; RUN: opt < %s --safe-stack -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare i32 @foo(i32* %p)
declare void @alloca_test_use([10 x i8]*)

define i32 @call_foo(i32* %a) safestack {
; CHECK-LABEL: @call_foo(
; CHECK-NEXT:    [[UNSAFE_STACK_PTR:%.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[UNSAFE_STACK_STATIC_TOP:%.*]] = getelementptr i8, i8* [[UNSAFE_STACK_PTR]], i32 -16
; CHECK-NEXT:    store i8* [[UNSAFE_STACK_STATIC_TOP]], i8** @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr i8, i8* [[UNSAFE_STACK_PTR]], i32 -10
; CHECK-NEXT:    [[X_UNSAFE:%.*]] = bitcast i8* [[TMP1]] to [10 x i8]*
; CHECK-NEXT:    call void @alloca_test_use([10 x i8]* [[X_UNSAFE]])
; CHECK-NEXT:    store i8* [[UNSAFE_STACK_PTR]], i8** @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[R:%.*]] = musttail call i32 @foo(i32* [[A:%.*]])
; CHECK-NEXT:    ret i32 [[R]]
;
  %x = alloca [10 x i8], align 1
  call void @alloca_test_use([10 x i8]* %x)
  %r = musttail call i32 @foo(i32* %a)
  ret i32 %r
}

define i32 @call_foo_cast(i32* %a) safestack {
; CHECK-LABEL: @call_foo_cast(
; CHECK-NEXT:    [[UNSAFE_STACK_PTR:%.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[UNSAFE_STACK_STATIC_TOP:%.*]] = getelementptr i8, i8* [[UNSAFE_STACK_PTR]], i32 -16
; CHECK-NEXT:    store i8* [[UNSAFE_STACK_STATIC_TOP]], i8** @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr i8, i8* [[UNSAFE_STACK_PTR]], i32 -10
; CHECK-NEXT:    [[X_UNSAFE:%.*]] = bitcast i8* [[TMP1]] to [10 x i8]*
; CHECK-NEXT:    call void @alloca_test_use([10 x i8]* [[X_UNSAFE]])
; CHECK-NEXT:    store i8* [[UNSAFE_STACK_PTR]], i8** @__safestack_unsafe_stack_ptr, align 8
; CHECK-NEXT:    [[R:%.*]] = musttail call i32 @foo(i32* [[A:%.*]])
; CHECK-NEXT:    [[T:%.*]] = bitcast i32 [[R]] to i32
; CHECK-NEXT:    ret i32 [[T]]
;
  %x = alloca [10 x i8], align 1
  call void @alloca_test_use([10 x i8]* %x)
  %r = musttail call i32 @foo(i32* %a)
  %t = bitcast i32 %r to i32
  ret i32 %t
}
