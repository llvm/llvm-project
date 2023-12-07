; RUN: opt < %s -passes=gvn -S | FileCheck %s

define i32 @main(ptr %p, i32 %x, i32 %y) {
block1:
  %z = load i32, ptr %p
  %cmp = icmp eq i32 %x, %y
	br i1 %cmp, label %block2, label %block3

block2:
 br label %block4

block3:
  %b = bitcast i32 0 to i32
  store i32 %b, ptr %p
  br label %block4

block4:
  %DEAD = load i32, ptr %p
  ret i32 %DEAD
}

; CHECK: define i32 @main(ptr %p, i32 %x, i32 %y) {
; CHECK-NEXT: block1:
; CHECK-NOT:    %z = load i32, ptr %p
; CHECK-NEXT:   %cmp = icmp eq i32 %x, %y
; CHECK-NEXT:   br i1 %cmp, label %block2, label %block3
; CHECK: block2:
; CHECK-NEXT:   %DEAD.pre = load i32, ptr %p
; CHECK-NEXT:   br label %block4
; CHECK: block3:
; CHECK-NEXT:   store i32 0, ptr %p
; CHECK-NEXT:   br label %block4
; CHECK: block4:
; CHECK-NEXT:   %DEAD = phi i32 [ 0, %block3 ], [ %DEAD.pre, %block2 ]
; CHECK-NEXT:   ret i32 %DEAD
; CHECK-NEXT: }
