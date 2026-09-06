; REQUIRES: asserts
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -debug-only=machine-scheduler -enable-aa-sched-mi -dag-limit-aa-calls=1 2>&1 | FileCheck %s --check-prefix=LIMIT1
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -debug-only=machine-scheduler -enable-aa-sched-mi -dag-limit-aa-calls=2 2>&1 | FileCheck %s --check-prefix=LIMIT2

define i32 @test(ptr noalias %a, i32 %v) {
  %ptr1 = getelementptr i32, ptr %a, i64 1
  %ptr2 = getelementptr i32, ptr %a, i64 500000
  %ptr3 = getelementptr i32, ptr %a, i64 1000000
  store i32 %v, ptr %ptr1, align 4
  %val1 = load i32, ptr %ptr2, align 4
  %val2 = load i32, ptr %ptr3, align 4
  %res = add i32 %val1, %val2
  ret i32 %res
}

; LIMIT1:      SU([[STORE:.*]]):   STRWui %{{.*}}, %{{.*}}, 1 :: (store (s32) into %ir.ptr1)
; LIMIT1-NEXT: # preds left
; LIMIT1-NEXT: # succs left       : 1
; LIMIT1:      Successors:
; LIMIT1-NEXT:  SU([[LOAD1:.*]]): Ord  Latency=1 Memory
; LIMIT1-NOT:   SU({{.*}}): Ord  Latency=1 Memory
; LIMIT1:      SU(3):

; LIMIT2:      SU([[STORE:.*]]):   STRWui %{{.*}}, %{{.*}}, 1 :: (store (s32) into %ir.ptr1)
; LIMIT2-NEXT: # preds left
; LIMIT2-NEXT: # succs left       : 0
; LIMIT2-NOT:  Successors:
; LIMIT2:      SU(3):
