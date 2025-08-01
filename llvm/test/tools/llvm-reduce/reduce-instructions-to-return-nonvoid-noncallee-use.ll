; Make sure we don't break on non-callee uses of funtions with a
; non-void return type.

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=instructions-to-return --test FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=RESULT %s < %t

; INTERESTING-LABEL: @interesting(
; INTERESTING: %inttoptr = inttoptr i64

; RESULT-LABEL: define ptr @interesting(i64 %arg) {
; RESULT-NEXT: %inttoptr = inttoptr i64 %arg to ptr
; RESULT-NEXT:  ret ptr %inttoptr
define void @interesting(i64 %arg) {
  %inttoptr = inttoptr i64 %arg to ptr
  %load = load i32, ptr %inttoptr
  ret void
}

declare i32 @func(ptr)

; RESULT-LABEL: define i32 @caller() {
; RESULT-NEXT: %call = call i32 @func(ptr @interesting)
; RESULT-NEXT: ret i32 %call
define void @caller() {
  %call = call i32 @func(ptr @interesting)
  ret void
}
