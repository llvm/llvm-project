; RUN: opt < %s -indirectbr-expand -S | FileCheck %s
; RUN: opt < %s -passes=indirectbr-expand -S | FileCheck %s
;
; REQUIRES: x86-registered-target

target triple = "x86_64-unknown-linux-gnu"

@test1.targets = constant [4 x ptr] [ptr blockaddress(@test1, %bb0),
                                     ptr blockaddress(@test1, %bb1),
                                     ptr blockaddress(@test1, %bb2),
                                     ptr blockaddress(@test1, %bb3)]
; CHECK-LABEL: @test1.targets = constant [4 x ptr]
; CHECK:       [ptr inttoptr (i64 1 to ptr),
; CHECK:        ptr inttoptr (i64 2 to ptr),
; CHECK:        ptr inttoptr (i64 3 to ptr),
; CHECK:        ptr blockaddress(@test1, %bb3)]

define void @test1(ptr readonly %p, ptr %sink) #0 {
; CHECK-LABEL: define void @test1(
entry:
  %i0 = load i64, ptr %p
  %target.i0 = getelementptr [4 x ptr], ptr @test1.targets, i64 0, i64 %i0
  %target0 = load ptr, ptr %target.i0
  ; Only a subset of blocks are viable successors here.
  indirectbr ptr %target0, [label %bb0, label %bb1]
; CHECK-NOT:     indirectbr
; CHECK:         %[[ENTRY_V:.*]] = ptrtoint ptr %{{.*}} to i64
; CHECK-NEXT:    br label %[[SWITCH_BB:.*]]

bb0:
  store volatile i64 0, ptr %sink
  br label %latch

bb1:
  store volatile i64 1, ptr %sink
  br label %latch

bb2:
  store volatile i64 2, ptr %sink
  br label %latch

bb3:
  store volatile i64 3, ptr %sink
  br label %latch

latch:
  %i.next = load i64, ptr %p
  %target.i.next = getelementptr [4 x ptr], ptr @test1.targets, i64 0, i64 %i.next
  %target.next = load ptr, ptr %target.i.next
  ; A different subset of blocks are viable successors here.
  indirectbr ptr %target.next, [label %bb1, label %bb2]
; CHECK-NOT:     indirectbr
; CHECK:         %[[LATCH_V:.*]] = ptrtoint ptr %{{.*}} to i64
; CHECK-NEXT:    br label %[[SWITCH_BB]]
;
; CHECK:       [[SWITCH_BB]]:
; CHECK-NEXT:    %[[V:.*]] = phi i64 [ %[[ENTRY_V]], %entry ], [ %[[LATCH_V]], %latch ]
; CHECK-NEXT:    switch i64 %[[V]], label %bb0 [
; CHECK-NEXT:      i64 2, label %bb1
; CHECK-NEXT:      i64 3, label %bb2
; CHECK-NEXT:    ]
}

attributes #0 = { "target-features"="+retpoline" }
