; RUN: opt < %s -indirectbr-expand -S | FileCheck %s
;
; REQUIRES: x86-registered-target

target triple = "x86_64-unknown-linux-gnu"

@test1.targets = constant [4 x i8*] [i8* blockaddress(@test1, %bb0),
                                     i8* blockaddress(@test1, %bb1),
                                     i8* blockaddress(@test1, %bb2),
                                     i8* blockaddress(@test1, %bb3)]
; CHECK-LABEL: @test1.targets = constant [4 x i8*]
; CHECK:       [i8* inttoptr (i64 1 to i8*),
; CHECK:        i8* inttoptr (i64 2 to i8*),
; CHECK:        i8* inttoptr (i64 3 to i8*),
; CHECK:        i8* blockaddress(@test1, %bb3)]

define void @test1(i64* readonly %p, i64* %sink) #0 {
; CHECK-LABEL: define void @test1(
entry:
  %i0 = load i64, i64* %p
  %target.i0 = getelementptr [4 x i8*], [4 x i8*]* @test1.targets, i64 0, i64 %i0
  %target0 = load i8*, i8** %target.i0
  ; Only a subset of blocks are viable successors here.
  indirectbr i8* %target0, [label %bb0, label %bb1]
; CHECK-NOT:     indirectbr
; CHECK:         %[[ENTRY_V:.*]] = ptrtoint i8* %{{.*}} to i64
; CHECK-NEXT:    br label %[[SWITCH_BB:.*]]

bb0:
  store volatile i64 0, i64* %sink
  br label %latch

bb1:
  store volatile i64 1, i64* %sink
  br label %latch

bb2:
  store volatile i64 2, i64* %sink
  br label %latch

bb3:
  store volatile i64 3, i64* %sink
  br label %latch

latch:
  %i.next = load i64, i64* %p
  %target.i.next = getelementptr [4 x i8*], [4 x i8*]* @test1.targets, i64 0, i64 %i.next
  %target.next = load i8*, i8** %target.i.next
  ; A different subset of blocks are viable successors here.
  indirectbr i8* %target.next, [label %bb1, label %bb2]
; CHECK-NOT:     indirectbr
; CHECK:         %[[LATCH_V:.*]] = ptrtoint i8* %{{.*}} to i64
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
