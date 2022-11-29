; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1(i1 %a, i1** %d) {
entry:
  %0 = load i1*, i1** %d, align 8
  br i1 %a, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %if.end

if.else:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %c.0 = phi i1 [ 1, %if.then ], [ 0, %if.else ]
  br i1 %c.0, label %if.then2, label %if.else3

if.then2:                                         ; preds = %if.end
  %rc = getelementptr inbounds i1, i1* %0, i64 0
  store i1 %c.0, i1* %rc, align 4
  br label %if.end6

if.else3:                                         ; preds = %if.end
  %rc5 = getelementptr inbounds i1, i1* %0, i64 0
  store i1 %c.0, i1* %rc5, align 4
  br label %if.end6

if.end6:                                          ; preds = %if.else3, %if.then2
  ret void
}

; CHECK-LABEL: define void @test1(
; CHECK:  %[[load:.*]] = load i1*, i1** %d, align 8
; CHECK:  %[[phi:.*]] = phi i1 [ true, {{.*}} ], [ false, {{.*}} ]

; CHECK: %[[gep0:.*]] = getelementptr inbounds i1, i1* %[[load]], i64 0
; CHECK: store i1 %[[phi]], i1* %[[gep0]], align 4

; Check that store instructions are hoisted.
; CHECK-NOT: store