; RUN: opt < %s -passes=gvn-sink -S | FileCheck %s

; Check that nonnull metadata for non-dominating loads is not propagated.
; CHECK-LABEL: @test1(
; CHECK-LABEL: if.end:
; CHECK:  %[[ptr:.*]] = phi i32**
; CHECK: %[[load:.*]] = load i32*, i32** %[[ptr]]
; CHECK-NOT: !nonnull
; CHECK: ret i32* %[[load]]
define i32* @test1(i1 zeroext %flag, i32*** %p) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %a = load i32**, i32*** %p
  %aa = load i32*, i32** %a, !nonnull !0
  br label %if.end

if.else:
  %b = load i32**, i32*** %p
  %bb= load i32*, i32** %b
  br label %if.end

if.end:
  %c = phi i32* [ %aa, %if.then ], [ %bb, %if.else ]
  ret i32* %c
}

; CHECK-LABEL: @test2(
; CHECK-LABEL: if.end:
; CHECK:  %[[ptr:.*]] = phi i32**
; CHECK: %[[load:.*]] = load i32*, i32** %[[ptr]]
; CHECK-NOT: !nonnull
; CHECK: ret i32* %[[load]]
define i32* @test2(i1 zeroext %flag, i32*** %p) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %a = load i32**, i32*** %p
  %aa = load i32*, i32** %a
  br label %if.end

if.else:
  %b = load i32**, i32*** %p
  %bb= load i32*, i32** %b, !nonnull !0
  br label %if.end

if.end:
  %c = phi i32* [ %aa, %if.then ], [ %bb, %if.else ]
  ret i32* %c
}


!0 = !{}
