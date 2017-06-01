; RUN: opt < %s -gvn -enable-pre -S | FileCheck %s

define i32 @main(i32 %p, i32 %q) {
block1:
    %cmp = icmp eq i32 %p, %q 
	br i1 %cmp, label %block2, label %block3

block2:
 %a = add i32 %p, 1
 br label %block4

block3:
  br label %block4
; CHECK: %.pre = add i32 %p, 1
; CHECK-NEXT: br label %block4

block4:
  %b = add i32 %p, 1
  ret i32 %b
; CHECK: %b.pre-phi = phi i32 [ %.pre, %block3 ], [ %a, %block2 ]
; CHECK-NEXT: ret i32 %b.pre-phi
}
