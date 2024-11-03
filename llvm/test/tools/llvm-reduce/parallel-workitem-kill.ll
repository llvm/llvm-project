; REQUIRES: thread_support
; RUN: llvm-reduce -j 4 %s -o %t --delta-passes=instructions --test %python --test-arg %S/Inputs/sleep-and-check-stores.py --test-arg 1 --test-arg 5
; RUN: FileCheck %s < %t

; CHECK: define void @foo
; CHECK: store
; CHECK: store
; CHECK: store
; CHECK: store
; CHECK: store
; CHECK: store
; CHECK-NEXT: ret void

define void @foo(ptr %ptr) {
  store i32 0, ptr %ptr
  store i32 1, ptr %ptr
  store i32 2, ptr %ptr
  store i32 3, ptr %ptr
  store i32 4, ptr %ptr
  store i32 5, ptr %ptr
  store i32 6, ptr %ptr
  store i32 7, ptr %ptr
  store i32 8, ptr %ptr
  store i32 9, ptr %ptr
  store i32 10, ptr %ptr
  store i32 11, ptr %ptr
  store i32 12, ptr %ptr
  store i32 13, ptr %ptr
  store i32 14, ptr %ptr
  store i32 15, ptr %ptr
  store i32 16, ptr %ptr
  ret void
}
