; REQUIRES: thread_support
; RUN: llvm-reduce --process-poll-interval=1 -j 4 %s -o %t --delta-passes=instructions --test %python --test-arg %S/Inputs/sleep.py --test-arg 2
; RUN: FileCheck %s < %t

; CHECK: define void @foo
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
  store i32 17, ptr %ptr
  store i32 18, ptr %ptr
  store i32 19, ptr %ptr
  store i32 20, ptr %ptr
  store i32 21, ptr %ptr
  store i32 22, ptr %ptr
  store i32 23, ptr %ptr
  store i32 24, ptr %ptr
  store i32 25, ptr %ptr
  store i32 26, ptr %ptr
  store i32 27, ptr %ptr
  store i32 28, ptr %ptr
  store i32 29, ptr %ptr
  store i32 30, ptr %ptr
  store i32 31, ptr %ptr
  store i32 32, ptr %ptr
  store i32 33, ptr %ptr
  store i32 34, ptr %ptr
  store i32 35, ptr %ptr
  store i32 36, ptr %ptr
  store i32 37, ptr %ptr
  store i32 38, ptr %ptr
  store i32 39, ptr %ptr
  store i32 40, ptr %ptr
  ret void
}

