; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1(i1 %a, ptr %d) {
entry:
  %0 = load ptr, ptr %d, align 8
  br i1 %a, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %if.end

if.else:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %c.0 = phi i1 [ 1, %if.then ], [ 0, %if.else ]
  br i1 %c.0, label %if.then2, label %if.else3

if.then2:                                         ; preds = %if.end
  store i1 %c.0, ptr %0, align 4
  br label %if.end6

if.else3:                                         ; preds = %if.end
  store i1 %c.0, ptr %0, align 4
  br label %if.end6

if.end6:                                          ; preds = %if.else3, %if.then2
  ret void
}

; CHECK-LABEL: define void @test1(
; CHECK:  %[[load:.*]] = load ptr, ptr %d, align 8
; CHECK:  %[[phi:.*]] = phi i1 [ true, {{.*}} ], [ false, {{.*}} ]

; CHECK: store i1 %[[phi]], ptr %[[load]], align 4

; Check that store instructions are hoisted.
; CHECK-NOT: store