; RUN: opt < %s -analyze -branch-prob -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -analyze -lazy-branch-prob -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

declare void @g(i32)

; Check correctness of reported probabilities in case of multiple edges between
; basic blocks. In this case sum of probabilities over all edges should be
; returned by BranchProbabilityInfo::getEdgeProbability.

define void @test1(i32 %x) {
;CHECK: edge entry -> return probability is 0x0ccccccd / 0x80000000 = 10.00%
;CHECK: edge entry -> bb0 probability is 0x26666666 / 0x80000000 = 30.00%
;CHECK: edge entry -> bb0 probability is 0x26666666 / 0x80000000 = 30.00%
;CHECK: edge entry -> bb0 probability is 0x26666666 / 0x80000000 = 30.00%
;CHECK: edge entry -> bb1 probability is 0x26666666 / 0x80000000 = 30.00%
;CHECK: edge entry -> bb1 probability is 0x26666666 / 0x80000000 = 30.00%
;CHECK: edge entry -> bb1 probability is 0x26666666 / 0x80000000 = 30.00%
;CHECK: edge entry -> bb2 probability is 0x26666666 / 0x80000000 = 30.00%
;CHECK: edge entry -> bb2 probability is 0x26666666 / 0x80000000 = 30.00%
;CHECK: edge entry -> bb2 probability is 0x26666666 / 0x80000000 = 30.00%
;CHECK: edge bb0 -> return probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge bb1 -> return probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge bb2 -> return probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

entry:
  switch i32 %x, label %return [
    i32 0, label %bb0
    i32 3, label %bb0
    i32 6, label %bb0
    i32 1, label %bb1
    i32 4, label %bb1
    i32 7, label %bb1
    i32 2, label %bb2
    i32 5, label %bb2
    i32 8, label %bb2
  ]

bb0:                                              ; preds = %entry, %entry, %entry
  tail call void @g(i32 0)
  br label %return

bb1:                                              ; preds = %entry, %entry, %entry
  tail call void @g(i32 1)
  br label %return

bb2:                                              ; preds = %entry, %entry, %entry
  tail call void @g(i32 2)
  br label %return

return:                                           ; preds = %bb2, %bb1, %bb0, %entry
  ret void
}
