; RUN: opt < %s -passes=gvn -S | FileCheck %s

; loop.then is not reachable from loop, so we should be able to deduce that the
; store through %phi2 cannot alias %ptr1.

; CHECK-LABEL: @test1
define void @test1(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: entry:
; CHECK: %[[GEP:.*]] = getelementptr inbounds i32, ptr %ptr1, i64 1
; CHECK: %[[VAL1:.*]] = load i32, ptr %[[GEP]]
entry:
  br label %loop.preheader

loop.preheader:
  %gep1 = getelementptr inbounds i32, ptr %ptr1, i64 1
  br label %loop

; CHECK-LABEL: loop:
; CHECK-NOT: load
loop:
  %phi1 = phi ptr [ %gep1, %loop.preheader ], [ %phi2, %loop.then ]
  %val1 = load i32, ptr %phi1
  br i1 false, label %loop.then, label %loop.if

loop.if:
  %gep2 = getelementptr inbounds i32, ptr %gep1, i64 1
  %val2 = load i32, ptr %gep2
  %cmp = icmp slt i32 %val1, %val2
  br label %loop.then

; CHECK-LABEL: loop.then
; CHECK: store i32 %[[VAL1]], ptr %phi2
loop.then:
  %phi2 = phi ptr [ %ptr2, %loop ], [ %gep2, %loop.if ]
  store i32 %val1, ptr %phi2
  store i32 0, ptr %ptr1
  br label %loop
}
