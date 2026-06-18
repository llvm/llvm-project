; RUN: opt -S < %s -passes='loop-unroll,load-store-vectorizer' -unroll-count=128 --capture-tracking-max-uses-to-explore=1024 | FileCheck %s

; Without using batching alias analysis, this test takes 6 seconds to compile. With, less than a second.
; This is because the mechanism that proves NoAlias in this case is very expensive (CaptureTracking.cpp),
; and caching the result leads to 2 calls to that mechanism instead of ~300,000 (run with -stats to see the difference)

; This test only demonstrates the compile time issue if capture-tracking-max-uses-to-explore is set to at least 1024,
; because with the default value of 100, the CaptureTracking analysis is not run, NoAlias is not proven, and the vectorizer gives up early.

@global_mem = external global i8, align 4

define void @compile-time-test() {
; CHECK-LABEL: define void @compile-time-test() {
; CHECK-COUNT-128: load <4 x i8>
entry:
  ; Create base pointer to a global variable with the inefficient pattern that Alias Analysis cannot easily traverse through.
  %global_base_loads = getelementptr i8, ptr inttoptr (i32 ptrtoint (ptr @global_mem to i32) to ptr), i64 0

  ; Create another pointer for the stores.
  %local_base_stores = alloca <512 x i8>, align 4

  ; 512 interwoven loads and stores in a loop that gets unrolled
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i_next, %loop ]

  %ptr_0 = getelementptr i8, ptr %global_base_loads, i64 %i
  %load_0 = load i8, ptr %ptr_0, align 4
  %ptr2_0 = getelementptr i8, ptr %local_base_stores, i64 %i
  store i8 %load_0, ptr %ptr2_0, align 4

  %i_1 = add i64 %i, 1

  %ptr_1 = getelementptr i8, ptr %global_base_loads, i64 %i_1
  %load_1 = load i8, ptr %ptr_1, align 1
  %ptr2_1 = getelementptr i8, ptr %local_base_stores, i64 %i_1
  store i8 %load_1, ptr %ptr2_1, align 1

  %i_2 = add i64 %i, 2

  %ptr_2 = getelementptr i8, ptr %global_base_loads, i64 %i_2
  %load_2 = load i8, ptr %ptr_2, align 2
  %ptr2_2 = getelementptr i8, ptr %local_base_stores, i64 %i_2
  store i8 %load_2, ptr %ptr2_2, align 2

  %i_3 = add i64 %i, 3

  %ptr_3 = getelementptr i8, ptr %global_base_loads, i64 %i_3
  %load_3 = load i8, ptr %ptr_3, align 1
  %ptr2_3 = getelementptr i8, ptr %local_base_stores, i64 %i_3
  store i8 %load_3, ptr %ptr2_3, align 1

  %i_next = add i64 %i, 4
  %cmp = icmp ult i64 %i_next, 512
  br i1 %cmp, label %loop, label %done

done:
  ret void
}
