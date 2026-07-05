; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=ifuncs --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-FINAL --input-file=%t %s


; CHECK-FINAL: @initialized_with_ifunc = global ptr @ifunc_constant_initializer_user
@initialized_with_ifunc = global ptr @ifunc_constant_initializer_user


; CHECK-FINAL: [[TABLE:@[0-9]+]] = internal global [[[TABLE_SIZE:[0-9]+]] x ptr] poison
; CHECK-FINAL: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 10, ptr @1, ptr null }]


; CHECK-INTERESTINGNESS: @ifunc_kept1 = ifunc

; CHECK-FINAL: @ifunc_kept1 = ifunc void (), ptr @resolver1
@ifunc_kept1 = ifunc void (), ptr @resolver1
@ifunc_removed2 = ifunc void (), ptr @resolver1

; CHECK-INTERESTINGNESS: @ifunc_kept3 =
; CHECK-FINAL: @ifunc_kept3 = ifunc i32 (double), ptr @resolver2
@ifunc_kept3 = ifunc i32 (double), ptr @resolver2


; Remove one with no users
@ifunc4_removed = ifunc float (i64), ptr @resolver2

; Keep one with no users
; CHECK-INTERESTINGNESS: @ifunc5_kept = ifunc
@ifunc5_kept = ifunc float (i64), ptr @resolver2


; Make sure the hidden is preserved
; CHECK-INTERESTINGNESS: @ifunc_kept_hidden =
; CHECK-FINAL: @ifunc_kept_hidden = hidden ifunc i32 (double), ptr @resolver3
@ifunc_kept_hidden = hidden ifunc i32 (double), ptr @resolver3
@ifunc7 = ifunc float (i64), ptr @resolver3

@ifunc_ptr_arg = ifunc void (ptr), ptr @resolver4


; CHECK-INTERESTINGNESS: @ifunc_nonvoid_kept0 = ifunc
@ifunc_nonvoid_kept0 = ifunc i32 (double), ptr @resolver5
@ifunc_nonvoid_removed0 = ifunc i32 (double), ptr @resolver5

; CHECK-INTERESTINGNESS: @ifunc_nonvoid_kept1 = ifunc
@ifunc_nonvoid_kept1 = ifunc i32 (double), ptr @resolver5
@ifunc_nonvoid_removed1 = ifunc i32 (double), ptr @resolver5

; CHECK-FINAL: @ifunc_constant_initializer_user = ifunc i32 (double), ptr @resolver5
@ifunc_constant_initializer_user = ifunc i32 (double), ptr @resolver5



define ptr @resolver1() {
  ret ptr inttoptr (i64 123 to ptr)
}

define ptr @resolver2() {
  ret ptr inttoptr (i64 456 to ptr)
}

define ptr @resolver3() {
  ret ptr inttoptr (i64 789 to ptr)
}

define ptr @resolver4() {
  ret ptr inttoptr (i64 999 to ptr)
}

define ptr @resolver5() {
  ret ptr inttoptr (i64 420 to ptr)
}

define void @call_ifunc_kept1() {
  ; CHECK-FINAL-LABEL: define void @call_ifunc_kept1() {
  ; CHECK-FINAL-NEXT: call void @ifunc_kept1()
  ; CHECK-FINAL-NEXT: ret void
  call void @ifunc_kept1()
  ret void
}

; Test call to removed ifunc
define void @call_ifunc_removed(ptr %ptr) {
  ; CHECK-FINAL-LABEL: define void @call_ifunc_removed(ptr %ptr)
  ; CHECK-FINAL-NEXT: %1 = load ptr, ptr @0, align 8
  ; CHECK-FINAL-NEXT:  call void %1()
  ; CHECK-FINAL-NEXT:  ret void
  call void @ifunc_removed2()
  ret void
}

; Test value use of removed ifunc
define void @store_ifunc_removed2(ptr %ptr) {
  ; CHECK-FINAL-LABEL: define void @store_ifunc_removed2(ptr %ptr) {
  ; CHECK-FINAL-NEXT: %1 = load ptr, ptr [[TABLE]], align 8
  ; CHECK-FINAL-NEXT: store ptr %1, ptr %ptr, align 8
  ; CHECK-FINAL-NEXT: %2 = load ptr, ptr @0, align 8
  ; CHECK-FINAL-NEXT: store ptr %ptr, ptr %2, align 8
  ; CHECK-FINAL-NEXT: ret void
  store ptr @ifunc_removed2, ptr %ptr
  store ptr %ptr, ptr @ifunc_removed2
  ret void
}

declare void @other_func(ptr)

; Check a call user, but not as the call operand
define void @call_ifunc_removed_is_argument(ptr %ptr) {
  ; CHECK-FINAL-LABEL: define void @call_ifunc_removed_is_argument(ptr %ptr) {
  ; CHECK-FINAL-NEXT: %1 = load ptr, ptr [[TABLE]], align 8
  ; CHECK-FINAL-NEXT: call void @other_func(ptr %1)
  ; CHECK-FINAL-NEXT: ret void
  call void @other_func(ptr @ifunc_removed2)
  ret void
}

; Check a call user calling the ifunc, and using the ifunc as an argument
define void @call_ifunc_removed_both_call_argument(ptr %ptr) {
  ; CHECK-FINAL-LABEL: define void @call_ifunc_removed_both_call_argument(
  ; CHECK-FINAL-NEXT: %1 = load ptr, ptr getelementptr inbounds ([[[TABLE_SIZE]] x ptr], ptr [[TABLE]], i32 0, i32 3), align 8
  ; CHECK-FINAL-NEXT: %2 = load ptr, ptr getelementptr inbounds ([[[TABLE_SIZE]] x ptr], ptr [[TABLE]], i32 0, i32 3), align 8
  ; CHECK-FINAL-NEXT: call void %1(ptr %1)
  ; CHECK-FINAL-NEXT: ret void
  call void @ifunc_ptr_arg(ptr @ifunc_ptr_arg)
  ret void
}

define i32 @call_ifunc_nonvoid(double %arg) {
  ; CHECK-FINAL-LABEL: define i32 @call_ifunc_nonvoid(double %arg) {
  ; CHECK-FINAL-NEXT: %ret0 = call i32 @ifunc_nonvoid_kept0(double %arg)
  ; CHECK-FINAL-NEXT: %1 = load ptr, ptr getelementptr inbounds ([[[TABLE_SIZE]] x ptr], ptr [[TABLE]], i32 0, i32 4), align 8
  ; CHECK-FINAL-NEXT: %ret1 = call i32 %1(double %arg)
  ; CHECK-FINAL-NEXT: %add = add i32 %ret0, %ret1
  ; CHECK-FINAL-NEXT: ret i32 %add
  %ret0 = call i32 @ifunc_nonvoid_kept0(double %arg)
  %ret1 = call i32 @ifunc_nonvoid_removed0(double %arg)
  %add = add i32 %ret0, %ret1
  ret i32 %add
}

; Use site is different than ifunc function type
define float @call_different_type_ifunc_nonvoid(double %arg) {
  ; CHECK-FINAL-LABEL: define float @call_different_type_ifunc_nonvoid(double %arg) {
  ; CHECK-FINAL-NEXT: %cast.arg = bitcast double %arg to i64
  ; CHECK-FINAL-NEXT: %ret0 = call float @ifunc_nonvoid_kept0(i64 %cast.arg)
  ; CHECK-FINAL-NEXT: %1 = load ptr, ptr getelementptr inbounds ([[[TABLE_SIZE]] x ptr], ptr [[TABLE]], i32 0, i32 4), align 8
  ; CHECK-FINAL-NEXT: %ret1 = call float %1(i64 %cast.arg)
  ; CHECK-FINAL-NEXT: %fadd = fadd float %ret0, %ret1
  ; CHECK-FINAL-NEXT: ret float %fadd
  %cast.arg = bitcast double %arg to i64
  %ret0 = call float(i64) @ifunc_nonvoid_kept0(i64 %cast.arg)
  %ret1 = call float(i64) @ifunc_nonvoid_removed0(i64 %cast.arg)
  %fadd = fadd float %ret0, %ret1
  ret float %fadd
}

; FIXME: Should be able to expand this, but we miss the call
; instruction in the constexpr cast.
define i32 @call_addrspacecast_callee_type_ifunc_nonvoid(double %arg) {
  ; CHECK-FINAL-LABEL: define i32 @call_addrspacecast_callee_type_ifunc_nonvoid(double %arg) {
  ; CHECK-FINAL-NEXT: %ret0 = call addrspace(1) i32 addrspacecast (ptr @ifunc_nonvoid_kept1 to ptr addrspace(1))(double %arg)
  ; CHECK-FINAL-NEXT: %ret1 = call addrspace(1) i32 addrspacecast (ptr @ifunc_nonvoid_removed1 to ptr addrspace(1))(double %arg)
  ; CHECK-FINAL-NEXT: %add = add i32 %ret0, %ret1
  ; CHECK-FINAL-NEXT: ret i32 %add
  %ret0 = call addrspace(1) i32 addrspacecast (ptr @ifunc_nonvoid_kept1 to ptr addrspace(1)) (double %arg)
  %ret1 = call addrspace(1) i32 addrspacecast (ptr @ifunc_nonvoid_removed1 to ptr addrspace(1)) (double %arg)
  %add = add i32 %ret0, %ret1
  ret i32 %add
}

define i32 @call_used_in_initializer(double %arg) {
  ; CHECK-FINAL-LABEL: define i32 @call_used_in_initializer(double %arg) {
  ; CHECK-FINAL-NEXT: %1 = load ptr, ptr getelementptr inbounds ([8 x ptr], ptr @0, i32 0, i32 7), align 8
  ; CHECK-FINAL-NEXT: %ret = call i32 %1(double %arg)
  ; CHECK-FINAL-NEXT: ret i32 %ret
  %ret = call i32 @ifunc_constant_initializer_user(double %arg)
  ret i32 %ret
}

; CHECK-FINAL-LABEL: define internal void @1() {
; CHECK-FINAL-NEXT: %1 = call ptr @resolver1()
; CHECK-FINAL-NEXT: store ptr %1, ptr @0, align 8
; CHECK-FINAL-NEXT: %2 = call ptr @resolver2()
; CHECK-FINAL-NEXT: store ptr %2, ptr getelementptr inbounds ([8 x ptr], ptr @0, i32 0, i32 1), align 8
; CHECK-FINAL-NEXT: %3 = call ptr @resolver3()
; CHECK-FINAL-NEXT: store ptr %3, ptr getelementptr inbounds ([8 x ptr], ptr @0, i32 0, i32 2), align 8
; CHECK-FINAL-NEXT: %4 = call ptr @resolver4()
; CHECK-FINAL-NEXT: store ptr %4, ptr getelementptr inbounds ([8 x ptr], ptr @0, i32 0, i32 3), align 8
; CHECK-FINAL-NEXT: %5 = call ptr @resolver5()
; CHECK-FINAL-NEXT: store ptr %5, ptr getelementptr inbounds ([8 x ptr], ptr @0, i32 0, i32 4), align 8
; CHECK-FINAL-NEXT: %6 = call ptr @resolver5()
; CHECK-FINAL-NEXT: store ptr %6, ptr getelementptr inbounds ([8 x ptr], ptr @0, i32 0, i32 5), align 8
; CHECK-FINAL-NEXT: %7 = call ptr @resolver5()
; CHECK-FINAL-NEXT: store ptr %7, ptr getelementptr inbounds ([8 x ptr], ptr @0, i32 0, i32 6), align 8
; CHECK-FINAL-NEXT: %8 = call ptr @resolver5()
; CHECK-FINAL-NEXT: store ptr %8, ptr getelementptr inbounds ([8 x ptr], ptr @0, i32 0, i32 7), align 8
; CHECK-FINAL-NEXT: ret void
; CHECK-FINAL-NEXT: }
