; RUN: opt < %s -gvn -S | FileCheck %s

; loop.then is not reachable from loop, so we should be able to deduce that the
; store through %phi2 cannot alias %ptr1.

; CHECK-LABEL: @test1
define void @test1(i32* %ptr1, i32* %ptr2) {
; CHECK-LABEL: entry:
; CHECK: %[[GEP:.*]] = getelementptr inbounds i32, i32* %ptr1, i64 1
; CHECK: %[[VAL1:.*]] = load i32, i32* %[[GEP]]
entry:
  br label %loop.preheader

loop.preheader:
  %gep1 = getelementptr inbounds i32, i32* %ptr1, i64 1
  br label %loop

; CHECK-LABEL: loop:
; CHECK-NOT: load
loop:
  %phi1 = phi i32* [ %gep1, %loop.preheader ], [ %phi2, %loop.then ]
  %val1 = load i32, i32* %phi1
  br i1 false, label %loop.then, label %loop.if

loop.if:
  %gep2 = getelementptr inbounds i32, i32* %gep1, i64 1
  %val2 = load i32, i32* %gep2
  %cmp = icmp slt i32 %val1, %val2
  br label %loop.then

; CHECK-LABEL: loop.then
; CHECK: store i32 %[[VAL1]], i32* %phi2
loop.then:
  %phi2 = phi i32* [ %ptr2, %loop ], [ %gep2, %loop.if ]
  store i32 %val1, i32* %phi2
  store i32 0, i32* %ptr1
  br label %loop
}
