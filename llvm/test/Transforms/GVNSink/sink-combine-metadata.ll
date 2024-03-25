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

; Check that nontemporal markings are propagated if both original stores are
; marked.
; CHECK-LABEL: @nontemporal(
; CHECK-LABEL: if.end:
; CHECK: !nontemporal
; CHECK: ret void
define void @nontemporal(i1 zeroext %flag, ptr %p) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %a = load ptr, ptr %p
  store ptr %p, ptr %a, align 8, !nontemporal !1
  br label %if.end

if.else:
  %b = load ptr, ptr %p
  store ptr %p, ptr %b, align 8, !nontemporal !1
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: @nontemporal_mismatch(
; CHECK-NOT: !nontemporal
define void @nontemporal_mismatch(i1 zeroext %flag, ptr %p) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %a = load ptr, ptr %p
  store ptr %p, ptr %a, align 8
  br label %if.end

if.else:
  %b = load ptr, ptr %p
  store ptr %p, ptr %b, align 8, !nontemporal !1
  br label %if.end

if.end:
  ret void
}

!0 = !{}
!1 = !{i32 1}
