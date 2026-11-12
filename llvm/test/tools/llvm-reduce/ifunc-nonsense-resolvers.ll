; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=ifuncs --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-FINAL --input-file=%t %s

; Check some cases that should probably be invalid IR don't break
; anything.

; CHECK-FINAL: @ifunc_with_arg = ifunc void (), ptr @resolver_with_arg
@ifunc_with_arg = ifunc void (), ptr @resolver_with_arg

define ptr @resolver_with_arg(i64 %arg) {
  %cast = inttoptr i64 %arg to ptr
  ret ptr %cast
}

; CHECK-INTERESTINGNESS: define void @call_with_arg()
define void @call_with_arg() {
  ; CHECK-FINAL: define void @call_with_arg() {
  ; CHECK-FINAL-NEXT: call void @ifunc_with_arg()
  ; CHECK-FINAL-NEXT: ret void
  call void @ifunc_with_arg()
  ret void
}
