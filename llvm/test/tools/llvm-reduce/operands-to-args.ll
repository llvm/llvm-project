; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --match-full-lines --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefixes=REDUCED,INTERESTING

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

; INTERESTING-LABEL: define void @func_caller() {
; REDUCED:             call void @func(i32 21, ptr null, ptr null)


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


define void @func_caller() {
entry:
  call void @func(i32 21)
  ret void
}


; Make sure to skip functions with non-direct call users
declare void @e(ptr)

; INTERESTING-LABEL: define void @g() {
define void @g() {
  call void @e(ptr @f)
  ret void
}

; INTERESTING-LABEL: define void @f(ptr %a) {
define void @f(ptr %a) {
  %1 = load ptr, ptr %a
  ret void
}
