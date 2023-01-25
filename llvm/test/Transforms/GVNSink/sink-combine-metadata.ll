; RUN: opt < %s -passes=gvn-sink -S | FileCheck %s

; Check that nonnull metadata for non-dominating loads is not propagated.
; CHECK-LABEL: @test1(
; CHECK-LABEL: if.end:
; CHECK:  %[[ptr:.*]] = phi ptr
; CHECK: %[[load:.*]] = load ptr, ptr %[[ptr]]
; CHECK-NOT: !nonnull
; CHECK: ret ptr %[[load]]
define ptr @test1(i1 zeroext %flag, ptr %p) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %a = load ptr, ptr %p
  %aa = load ptr, ptr %a, !nonnull !0
  br label %if.end

if.else:
  %b = load ptr, ptr %p
  %bb= load ptr, ptr %b
  br label %if.end

if.end:
  %c = phi ptr [ %aa, %if.then ], [ %bb, %if.else ]
  ret ptr %c
}

; CHECK-LABEL: @test2(
; CHECK-LABEL: if.end:
; CHECK:  %[[ptr:.*]] = phi ptr
; CHECK: %[[load:.*]] = load ptr, ptr %[[ptr]]
; CHECK-NOT: !nonnull
; CHECK: ret ptr %[[load]]
define ptr @test2(i1 zeroext %flag, ptr %p) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %a = load ptr, ptr %p
  %aa = load ptr, ptr %a
  br label %if.end

if.else:
  %b = load ptr, ptr %p
  %bb= load ptr, ptr %b, !nonnull !0
  br label %if.end

if.end:
  %c = phi ptr [ %aa, %if.then ], [ %bb, %if.else ]
  ret ptr %c
}


!0 = !{}
