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

define internal i1 @f1(i32 %x, i32 %y, i1 %cmp) {
; CHECK-LABEL: define internal i1 @f1(i32 %x, i32 %y, i1 %cmp) {
; CHECK-NEXT:    %sel.1 = select i1 %cmp, i32 %x, i32 %y
; CHECK-NEXT:    %c.2 = icmp sgt i32 %sel.1, 100
; CHECK-NEXT:    %c.3 = icmp eq i32 %sel.1, 50
; CHECK-NEXT:    %res.1 = add i1 false, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, %c.3
; CHECK-NEXT:    %res.3 = add i1 %res.2, false
; CHECK-NEXT:    ret i1 %res.3
;
  %sel.1 = select i1 %cmp, i32 %x, i32 %y
  %c.1 = icmp sgt i32 %sel.1, 300
  %c.2 = icmp sgt i32 %sel.1, 100
  %c.3 = icmp eq i32 %sel.1, 50
  %c.4 = icmp slt i32 %sel.1, 9
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  ret i1 %res.3
}

define i1 @caller1(i1 %cmp) {
; CHECK-LABEL:  define i1 @caller1(i1 %cmp) {
; CHECK-NEXT:    %call.1 = tail call i1 @f1(i32 10, i32 100, i1 %cmp)
; CHECK-NEXT:    %call.2 = tail call i1 @f1(i32 20, i32 200, i1 %cmp)
; CHECK-NEXT:    %res = and i1 %call.1, %call.2
; CHECK-NEXT:    ret i1 %res
;
  %call.1 = tail call i1 @f1(i32 10, i32 100, i1 %cmp)
  %call.2 = tail call i1 @f1(i32 20, i32 200, i1 %cmp)
  %res = and i1 %call.1, %call.2
  ret i1 %res
}


define i1 @f2(i32 %x, i32 %y, i1 %cmp) {
; CHECK-LABEL: define i1 @f2(i32 %x, i32 %y, i1 %cmp) {
; CHECK-NEXT:    %sel.1 = select i1 %cmp, i32 %x, i32 %y
; CHECK-NEXT:    %c.1 = icmp sgt i32 %sel.1, 300
; CHECK-NEXT:    %c.2 = icmp sgt i32 %sel.1, 100
; CHECK-NEXT:    %c.3 = icmp eq i32 %sel.1, 50
; CHECK-NEXT:    %c.4 = icmp slt i32 %sel.1, 9
; CHECK-NEXT:    %res.1 = add i1 %c.1, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, %c.3
; CHECK-NEXT:    %res.3 = add i1 %res.2, %c.4
; CHECK-NEXT:    ret i1 %res.3
;
  %sel.1 = select i1 %cmp, i32 %x, i32 %y
  %c.1 = icmp sgt i32 %sel.1, 300
  %c.2 = icmp sgt i32 %sel.1, 100
  %c.3 = icmp eq i32 %sel.1, 50
  %c.4 = icmp slt i32 %sel.1, 9
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  ret i1 %res.3
}

define i1 @caller2(i32 %y, i1 %cmp) {
; CHECK-LABEL:  define i1 @caller2(i32 %y, i1 %cmp) {
; CHECK-NEXT:    %call.1 = tail call i1 @f2(i32 10, i32 %y, i1 %cmp)
; CHECK-NEXT:    %call.2 = tail call i1 @f2(i32 20, i32 %y, i1 %cmp)
; CHECK-NEXT:    %res = and i1 %call.1, %call.2
; CHECK-NEXT:    ret i1 %res
;
  %call.1 = tail call i1 @f2(i32 10, i32 %y, i1 %cmp)
  %call.2 = tail call i1 @f2(i32 20, i32 %y, i1 %cmp)
  %res = and i1 %call.1, %call.2
  ret i1 %res
}

@GV = common global i32 0, align 4

define i32 @f3_constantexpr_cond(i32 %x, i32 %y) {
; CHECK-LABEL: define i32 @f3_constantexpr_cond(i32 %x, i32 %y)
; CHECK-NEXT:   %sel.1 = select i1 icmp eq (i32* bitcast (i32 (i32, i32)* @f3_constantexpr_cond to i32*), i32* @GV), i32 %x, i32 %y
; CHECK-NEXT:   ret i32 %sel.1
;
  %sel.1 = select i1 icmp eq (i32* bitcast (i32 (i32, i32)* @f3_constantexpr_cond to i32*), i32* @GV), i32 %x, i32 %y
  ret i32 %sel.1
}

define i32 @caller3(i32 %y) {
; CHECK-LABEL:  define i32 @caller3(i32 %y) {
; CHECK-NEXT:    %call.1 = tail call i32 @f3_constantexpr_cond(i32 10, i32 %y)
; CHECK-NEXT:    %call.2 = tail call i32 @f3_constantexpr_cond(i32 20, i32 %y)
; CHECK-NEXT:    %res = and i32 %call.1, %call.2
; CHECK-NEXT:    ret i32 %res
;
  %call.1 = tail call i32 @f3_constantexpr_cond(i32 10, i32 %y)
  %call.2 = tail call i32 @f3_constantexpr_cond(i32 20, i32 %y)
  %res = and i32 %call.1, %call.2
  ret i32 %res
}
