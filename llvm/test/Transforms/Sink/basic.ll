; RUN: opt < %s -basicaa -sink -S | FileCheck %s

@A = external global i32
@B = external global i32

; Sink should sink the load past the store (which doesn't overlap) into
; the block that uses it.

;      CHECK: @foo
;      CHECK: true:
; CHECK-NEXT: %l = load i32* @A
; CHECK-NEXT: ret i32 %l

define i32 @foo(i1 %z) {
  %l = load i32* @A
  store i32 0, i32* @B
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}

; But don't sink load volatiles...

;      CHECK: @foo2
;      CHECK: load volatile
; CHECK-NEXT: store i32

define i32 @foo2(i1 %z) {
  %l = load volatile i32* @A
  store i32 0, i32* @B
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}

; Sink to the nearest post-dominator

;      CHECK: @diamond
;      CHECK: X:
; CHECK-NEXT: phi
; CHECK-NEXT: mul nsw
; CHECK-NEXT: sub

define i32 @diamond(i32 %a, i32 %b, i32 %c) {
  %1 = mul nsw i32 %c, %b
  %2 = icmp sgt i32 %a, 0
  br i1 %2, label %B0, label %B1

B0:                                       ; preds = %0
  br label %X

B1:                                      ; preds = %0
  br label %X

X:                                     ; preds = %5, %3
  %.01 = phi i32 [ %c, %B0 ], [ %a, %B1 ]
  %R = sub i32 %1, %.01
  ret i32 %R
}

