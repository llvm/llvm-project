; RUN: opt < %s -sccp -S | FileCheck %s

define i32 @test1(i1 %C) {
; CHECK-LABEL: define i32 @test1(
; CHECK-NEXT:   ret i32 0
;
	%X = select i1 %C, i32 0, i32 0		; <i32> [#uses=1]
	ret i32 %X
}

define i32 @test2(i1 %C) {
; CHECK-LABEL: define i32 @test2(
; CHECK-NEXT:   ret i32 0
;
	%X = select i1 %C, i32 0, i32 undef		; <i32> [#uses=1]
	ret i32 %X
}

define i1 @f2(i32 %x, i1 %cmp) {
; CHECK-LABEL: define i1 @f2(i32 %x, i1 %cmp) {
; CHECK-NEXT:    %sel.1 = select i1 %cmp, i32 %x, i32 10
; CHECK-NEXT:    %c.1 = icmp sgt i32 %sel.1, 300
; CHECK-NEXT:    %c.2 = icmp sgt i32 %sel.1, 100
; CHECK-NEXT:    %c.3 = icmp eq i32 %sel.1, 50
; CHECK-NEXT:    %c.4 = icmp slt i32 %sel.1, 9
; CHECK-NEXT:    %res.1 = add i1 %c.1, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, %c.3
; CHECK-NEXT:    %res.3 = add i1 %res.2, %c.4
; CHECK-NEXT:    ret i1 %res.3
;
  %sel.1 = select i1 %cmp, i32 %x, i32 10
  %c.1 = icmp sgt i32 %sel.1, 300
  %c.2 = icmp sgt i32 %sel.1, 100
  %c.3 = icmp eq i32 %sel.1, 50
  %c.4 = icmp slt i32 %sel.1, 9
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  ret i1 %res.3
}
