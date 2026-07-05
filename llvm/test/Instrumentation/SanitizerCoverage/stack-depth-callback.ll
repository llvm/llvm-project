; This check verifies that stack depth callback instrumentation works correctly.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-stack-depth -sanitizer-coverage-stack-depth-callback-min=1 -S | FileCheck %s --check-prefixes=COMMON,CB1
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-stack-depth -sanitizer-coverage-stack-depth-callback-min=8 -S | FileCheck %s --check-prefixes=COMMON,CB8
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-stack-depth -sanitizer-coverage-stack-depth-callback-min=16 -S | FileCheck %s --check-prefixes=COMMON,CB16
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-stack-depth -sanitizer-coverage-stack-depth-callback-min=32 -S | FileCheck %s --check-prefixes=COMMON,CB32
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-stack-depth -sanitizer-coverage-stack-depth-callback-min=64 -S | FileCheck %s --check-prefixes=COMMON,CB64
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-stack-depth -sanitizer-coverage-stack-depth-callback-min=128 -S | FileCheck %s --check-prefixes=COMMON,CB128

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; No stack, just return: our leaf function
define i32 @foo() {
; COMMON-LABEL: define i32 @foo() {
; COMMON-NEXT:  entry:
; CB1-NOT:        call void @__sanitizer_cov_stack_depth()
; CB8-NOT:        call void @__sanitizer_cov_stack_depth()
; CB16-NOT:       call void @__sanitizer_cov_stack_depth()
; CB32-NOT:       call void @__sanitizer_cov_stack_depth()
; CB64-NOT:       call void @__sanitizer_cov_stack_depth()
; CB128-NOT:      call void @__sanitizer_cov_stack_depth()
; COMMON-NEXT:    ret i32 7
;
entry:

  ret i32 7
}

; No stack, just function call
define i32 @retcall() {
; COMMON-LABEL: define i32 @retcall() {
; COMMON-NEXT:  entry:
; CB1-NOT:        call void @__sanitizer_cov_stack_depth()
; CB8-NOT:        call void @__sanitizer_cov_stack_depth()
; CB16-NOT:       call void @__sanitizer_cov_stack_depth()
; CB32-NOT:       call void @__sanitizer_cov_stack_depth()
; CB64-NOT:       call void @__sanitizer_cov_stack_depth()
; CB128-NOT:      call void @__sanitizer_cov_stack_depth()
; COMMON-NEXT:    [[CALL:%.*]] = call i32 @foo()
; COMMON-NEXT:    ret i32 [[CALL]]
entry:

  %call = call i32 @foo()
  ret i32 %call
}

; No stack, just function call, with argument
define i32 @witharg(i32 %input) {
; COMMON-LABEL: define i32 @witharg(i32 %input) {
; COMMON-NEXT:  entry:
; CB1-NOT:        call void @__sanitizer_cov_stack_depth()
; CB8-NOT:        call void @__sanitizer_cov_stack_depth()
; CB16-NOT:       call void @__sanitizer_cov_stack_depth()
; CB32-NOT:       call void @__sanitizer_cov_stack_depth()
; CB64-NOT:       call void @__sanitizer_cov_stack_depth()
; CB128-NOT:      call void @__sanitizer_cov_stack_depth()
; COMMON-NEXT:    [[CALL:%.*]] = call i32 @foo()
; COMMON-NEXT:    ret i32 [[CALL]]
entry:

  %call = call i32 @foo()
  ret i32 %call
}

; 4 byte stack of scalars
define i32 @alloc4_0() {
; COMMON-LABEL: define i32 @alloc4_0() {
; COMMON-NEXT:  entry:
; COMMON-NEXT:    [[VAR:%.*]] = alloca i32, align 4
; CB1-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB8-NOT:        call void @__sanitizer_cov_stack_depth()
; CB16-NOT:       call void @__sanitizer_cov_stack_depth()
; CB32-NOT:       call void @__sanitizer_cov_stack_depth()
; CB64-NOT:       call void @__sanitizer_cov_stack_depth()
; CB128-NOT:      call void @__sanitizer_cov_stack_depth()
; COMMON-NEXT:    [[CALL:%.*]] = call i32 @foo()
; COMMON-NEXT:    ret i32 [[CALL]]
entry:
  %var1 = alloca i32, align 4

  %call = call i32 @foo()
  ret i32 %call
}

; 16 byte stack of scalars
define i32 @alloc16_0() {
; COMMON-LABEL: define i32 @alloc16_0() {
; COMMON-NEXT:  entry:
; COMMON-NEXT:    [[VAR:%.*]] = alloca i32, align 4
; COMMON-NEXT:    [[VAR:%.*]] = alloca i32, align 4
; COMMON-NEXT:    [[VAR:%.*]] = alloca i32, align 4
; COMMON-NEXT:    [[VAR:%.*]] = alloca i32, align 4
; CB1-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB8-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB16-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB32-NOT:       call void @__sanitizer_cov_stack_depth()
; CB64-NOT:       call void @__sanitizer_cov_stack_depth()
; CB128-NOT:      call void @__sanitizer_cov_stack_depth()
; COMMON-NEXT:    [[CALL:%.*]] = call i32 @foo()
; COMMON-NEXT:    ret i32 [[CALL]]
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  %var3 = alloca i32, align 4
  %var4 = alloca i32, align 4

  %call = call i32 @foo()
  ret i32 %call
}

; 32 byte stack of scalars
define i32 @alloc32_0() {
; COMMON-LABEL: define i32 @alloc32_0() {
; COMMON-NEXT:  entry:
; COMMON-NEXT:    [[VAR:%.*]] = alloca i64, align 8
; COMMON-NEXT:    [[VAR:%.*]] = alloca i64, align 8
; COMMON-NEXT:    [[VAR:%.*]] = alloca i64, align 8
; COMMON-NEXT:    [[VAR:%.*]] = alloca i64, align 8
; CB1-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB8-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB16-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB32-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB64-NOT:       call void @__sanitizer_cov_stack_depth()
; CB128-NOT:      call void @__sanitizer_cov_stack_depth()
; COMMON-NEXT:    [[CALL:%.*]] = call i32 @foo()
; COMMON-NEXT:    ret i32 [[CALL]]
entry:
  %var1 = alloca i64, align 8
  %var2 = alloca i64, align 8
  %var3 = alloca i64, align 8
  %var4 = alloca i64, align 8

  %call = call i32 @foo()
  ret i32 %call
}

; 36 byte stack of 1 4 byte scalar and 1 32 byte array
define i32 @alloc4_32x1() {
; COMMON-LABEL: define i32 @alloc4_32x1() {
; COMMON-NEXT:  entry:
; COMMON-NEXT:    [[VAR:%.*]] = alloca i8, i32 32, align 4
; COMMON-NEXT:    [[VAR:%.*]] = alloca i32, align 4
; CB1-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB8-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB16-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB32-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB64-NOT:       call void @__sanitizer_cov_stack_depth()
; CB128-NOT:      call void @__sanitizer_cov_stack_depth()
; COMMON-NEXT:    [[CALL:%.*]] = call i32 @foo()
; COMMON-NEXT:    ret i32 [[CALL]]
entry:
  %stack_array1 = alloca i8, i32 32, align 4
  %var1 = alloca i32, align 4

  %call = call i32 @foo()
  ret i32 %call
}

; 64 byte stack of 2 32 byte arrays
define i32 @alloc0_32x2() {
; COMMON-LABEL: define i32 @alloc0_32x2() {
; COMMON-NEXT:  entry:
; COMMON-NEXT:    [[VAR:%.*]] = alloca i8, i32 32, align 4
; COMMON-NEXT:    [[VAR:%.*]] = alloca i8, i32 32, align 4
; CB1-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB8-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB16-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB32-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB64-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB128-NOT:      call void @__sanitizer_cov_stack_depth()
; COMMON-NEXT:    [[CALL:%.*]] = call i32 @foo()
; COMMON-NEXT:    ret i32 [[CALL]]
entry:
  %stack_array1 = alloca i8, i32 32, align 4
  %stack_array2 = alloca i8, i32 32, align 4

  %call = call i32 @foo()
  ret i32 %call
}

; 64 byte stack of 1 64 byte array
define i32 @alloc0_64x1() {
; COMMON-LABEL: define i32 @alloc0_64x1() {
; COMMON-NEXT:  entry:
; COMMON-NEXT:    [[VAR:%.*]] = alloca i8, i32 64, align 4
; CB1-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB8-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB16-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB32-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB64-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB128-NOT:      call void @__sanitizer_cov_stack_depth()
; COMMON-NEXT:    [[CALL:%.*]] = call i32 @foo()
; COMMON-NEXT:    ret i32 [[CALL]]
entry:
  %stack_array = alloca i8, i32 64, align 4

  %call = call i32 @foo()
  ret i32 %call
}

; dynamic stack sized by i32
define i32 @alloc0_32xDyn(i32 %input) {
; COMMON-LABEL: define i32 @alloc0_32xDyn(i32 %input) {
; COMMON-NEXT:  entry:
; COMMON-NEXT:    [[VAR:%.*]] = alloca i8, i32 %input, align 4
; CB1-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB8-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB16-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB32-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB64-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB128-NEXT:     call void @__sanitizer_cov_stack_depth()
; COMMON-NEXT:    [[CALL:%.*]] = call i32 @foo()
; COMMON-NEXT:    ret i32 [[CALL]]
entry:
  %stack_array1 = alloca i8, i32 %input, align 4

  %call = call i32 @foo()
  ret i32 %call
}

; true dynamic stack sized by i32, from C:
; static int dyamic_alloca(int size)
; {
;   int array[size];
;   return foo();
; }
define dso_local i32 @dynamic_alloca(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store i32 %0, ptr %2, align 4
  %5 = load i32, ptr %2, align 4
  %6 = zext i32 %5 to i64
; COMMON-LABEL:   %7 = call ptr @llvm.stacksave
; COMMON-NEXT:    store ptr %7, ptr %3, align 8
; COMMON-NEXT:    [[VAR:%.*]] = alloca i32, i64 %6, align 16
; CB1-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB8-NEXT:       call void @__sanitizer_cov_stack_depth()
; CB16-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB32-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB64-NEXT:      call void @__sanitizer_cov_stack_depth()
; CB128-NEXT:     call void @__sanitizer_cov_stack_depth()
  %7 = call ptr @llvm.stacksave.p0()
  store ptr %7, ptr %3, align 8
  %8 = alloca i32, i64 %6, align 16
  store i64 %6, ptr %4, align 8
  %9 = call i32 @foo()
  %10 = load ptr, ptr %3, align 8
; COMMON-LABEL: call void @llvm.stackrestore
; COMMON-NEXT: ret i32 %9
  call void @llvm.stackrestore.p0(ptr %10)
  ret i32 %9
}
