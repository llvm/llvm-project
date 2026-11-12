; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --match-full-lines --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefixes=REDUCED,INTERESTING

; REDUCED-LABEL: define void @func(i32 %k, i32 %extra.arg, ptr %Global, ptr %Local) {

; Keep one reference to the original value.
; INTERESTING: %[[LOCAL:Local[0-9]*]] = alloca i32, align 4

; Everything else must use the function argument.
; REDUCED: store i32 21, ptr %Global, align 4
; REDUCED: store i32 0, ptr %Local, align 4
; REDUCED: store i32 0, ptr %Global, align 4

; Do not add any arguments for %Keep and @GlobalKeep.
; INTERESTING: %[[KEEP:LocalKeep[0-9]*]] = add i32 %k, 21
; INTERESTING: store i32 %[[KEEP]], ptr @GlobalKeep, align 4

; Do not add any arguments if the call type was already mismatched

; INTERESTING-LABEL: define void @mismatched_func_caller() {
; REDUCED:             call void @func(i32 21)

@Global = global i32 42
@GlobalKeep = global i32 42

define void @func(i32 %k, i32 %extra.arg) {
entry:
  %Local = alloca i32, align 4
  store i32 21, ptr @Global, align 4
  store i32 0, ptr %Local, align 4
  store i32 0, ptr @Global, align 4
  %LocalKeep = add i32 %k, 21
  store i32 %LocalKeep, ptr @GlobalKeep, align 4
  ret void
}

; This call has the wrong signature for the original underlying call,
; so getCalledFunction does not return a reference to the function.
define void @mismatched_func_caller() {
entry:
  call void @func(i32 21)
  ret void
}

