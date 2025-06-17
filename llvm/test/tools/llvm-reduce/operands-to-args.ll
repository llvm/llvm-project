; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefix=REDUCED

; INTERESTING-LABEL: define void @func(
; REDUCED-LABEL: define void @func(i32 %k, ptr %Local, ptr %Global) {

; Keep one reference to the original value.
; INTERESTING: %[[LOCAL:Local[0-9]*]] = alloca i32, align 4
; INTERESTING: store i32 42, ptr %[[LOCAL]], align 4
; INTERESTING: store i32 42, ptr @Global, align 4

; Everything else must use the function argument.
; REDUCED: store i32 21, ptr %Local, align 4
; REDUCED: store i32 21, ptr %Global, align 4
; REDUCED: store i32 0, ptr %Local, align 4
; REDUCED: store i32 0, ptr %Global, align 4
; REDUCED: store float 0.000000e+00, ptr %Global, align 4

; Do not add any arguments for %Keep and @GlobalKeep.
; INTERESTING: %[[KEEP:LocalKeep[0-9]*]] = add i32 %k, 21
; INTERESTING: store i32 %[[KEEP]], ptr @GlobalKeep, align 4


@Global = global i32 42
@GlobalKeep = global i32 42

define void @func(i32 %k) {
entry:
  %Local = alloca i32, align 4

  store i32 42, ptr %Local, align 4
  store i32 42, ptr @Global, align 4

  store i32 21, ptr %Local, align 4
  store i32 21, ptr @Global, align 4

  store i32 0, ptr %Local, align 4
  store i32 0, ptr @Global, align 4

  store float 0.000000e+00, ptr @Global, align 4

  %LocalKeep = add i32 %k, 21
  store i32 %LocalKeep, ptr @GlobalKeep, align 4

  ret void
}

; INTERESTING-LABEL: define void @func_caller(
; REDUCED-LABEL: define void @func_caller() {
; REDUCED: call void @func(i32 21, ptr null, ptr null)
define void @func_caller() {
entry:
  call void @func(i32 21)
  ret void
}


; Make sure to skip functions with non-direct call users
declare void @e(ptr)

; INTERESTING-LABEL: define void @g(
; REDUCED-LABEL: define void @g(ptr %f) {
; REDUCED: call void @e(ptr %f)
define void @g() {
  call void @e(ptr @f)
  ret void
}

; INTERESTING-LABEL: define void @f(
; REDUCED-LABEL: define void @f(ptr %a) {
; REDUCED: %1 = load ptr, ptr %a, align 8
define void @f(ptr %a) {
  %1 = load ptr, ptr %a
  ret void
}

@gv_init_use = global [1 x ptr] [ptr @has_global_init_user]

; INTERESTING-LABEL: define void @has_global_init_user(
; REDUCED-LABEL: define void @has_global_init_user(ptr %Local) {
define void @has_global_init_user() {
  %Local = alloca i32, align 4
  store i32 42, ptr %Local, align 4
  ret void
}

; INTERESTING-LABEL: define void @has_callee_and_arg_user(
; REDUCED-LABEL: define void @has_callee_and_arg_user(ptr %orig.arg, ptr %Local) {
define void @has_callee_and_arg_user(ptr %orig.arg) {
  %Local = alloca i32, align 4
  store i32 42, ptr %Local, align 4
  ret void
}

declare void @ptr_user(ptr)

; INTERESTING-LABEL: define void @calls_and_passes_func(
; REDUCED-LABEL: define void @calls_and_passes_func(ptr %has_callee_and_arg_user) {
; REDUCED: call void @has_callee_and_arg_user(ptr %has_callee_and_arg_user, ptr null)
define void @calls_and_passes_func() {
  call void @has_callee_and_arg_user(ptr @has_callee_and_arg_user)
  ret void
}

; INTERESTING-LABEL: define void @has_wrong_callsite_type_user(
; REDUCED-LABEL: define void @has_wrong_callsite_type_user(i32 %extra.arg, ptr %Local) {
define void @has_wrong_callsite_type_user(i32 %extra.arg) {
  %Local = alloca i32, align 4
  store i32 42, ptr %Local, align 4
  ret void
}

; INTERESTING-LABEL: define void @calls_wrong_func_type(
; REDUCED: call void @has_wrong_callsite_type_user()
define void @calls_wrong_func_type() {
  call void @has_wrong_callsite_type_user()
  ret void
}
