; REQUIRES: have_tflite
; REQUIRES: x86_64-linux
;
; Check that we can log more than 1 function.
;
; RUN: llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-advisor=development \
; RUN:   -regalloc-training-log=%t1 < %s
; RUN: FileCheck --input-file %t1 %s

; RUN: llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-priority-advisor=development \
; RUN:   -regalloc-priority-training-log=%t2 < %s
; RUN: FileCheck --input-file %t2 %s

declare void @f();

define void @f1(i64 %lhs, i64 %rhs, i64* %addr) {
  ret void
}

define void @f2(i64 %lhs, i64 %rhs, i64* %addr) {
  %sum = add i64 %lhs, %rhs
  call void @f();
  store i64 %sum, i64* %addr
  ret void
}

define void @f3(i64 %lhs, i64 %rhs, i64* %addr) {
  ret void
}

; CHECK-NOT:  {"context":"f1"}
; CHECK:      {"context":"f2"}
; CHECK-NOT:  {"context":"f3"}
