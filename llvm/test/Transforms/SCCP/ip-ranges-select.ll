; RUN: opt -ipsccp -S %s -o -| FileCheck %s

define void @caller.1(i8* %arg) {
; CHECK-LABEL: define void @caller.1(i8* %arg) {
; CHECK-NEXT:    %r.1 = tail call i32 @callee.1(i32 4)
; CHECK-NEXT:    %r.2 = tail call i32 @callee.1(i32 2)
; CHECK-NEXT:    call void @use(i32 20)
; CHECK-NEXT:    ret void
;
  %r.1 = tail call i32 @callee.1(i32 4)
  %r.2 = tail call i32 @callee.1(i32 2)
  %r.3 = add i32 %r.1, %r.2
  call void @use(i32 %r.3)
  ret void
}

define internal i32 @callee.1(i32 %arg) {
; CHECK-LABEL: define internal i32 @callee.1(i32 %arg) {
; CHECK-NEXT:    %sel = select i1 false, i32 16, i32 %arg
; CHECK-NEXT:    br label %bb10
;
; CHECK-LABEL: bb10:
; CHECK-NEXT:    ret i32 undef
;
  %c.1 = icmp slt i32 %arg, 0
  %sel = select i1 %c.1, i32 16, i32 %arg
  %c.2 = icmp eq i32 %sel, 0
  br i1 %c.2, label %bb12, label %bb10

bb10:                                             ; preds = %bb8
  ret i32 10

bb12:                                             ; preds = %bb8, %bb3, %bb
  ret i32 12
}

declare void @use(i32)
