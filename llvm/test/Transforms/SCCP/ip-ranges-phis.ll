; RUN: opt < %s -passes=ipsccp -S | FileCheck %s

define internal i32 @f1(i32 %x) {
; CHECK-LABEL: define internal i32 @f1(
; CHECK-NEXT:    ret i32 undef
;
  %cmp = icmp sgt i32 %x, 300
  %res = select i1 %cmp, i32 1, i32 2
  ret i32 %res
}

; %res is a constant range [0, 2) from a PHI node.
define i32 @caller1(i1 %cmp) {
; CHECK-LABEL: define i32 @caller1(
; CHECK-LABEL: entry:
; CHECK-NEXT:    br i1 %cmp, label %if.true, label %end

; CHECK-LABEL: if.true:
; CHECK-NEXT:    br label %end

; CHECK-LABEL: end:
; CHECK-NEXT:    %res = phi i32 [ 0, %entry ], [ 1, %if.true ]
; CHECK-NEXT:    %call1 = tail call i32 @f1(i32 %res)
; CHECK-NEXT:    ret i32 2
;
entry:
  br i1 %cmp, label %if.true, label %end

if.true:
  br label %end

end:
  %res = phi i32 [ 0, %entry], [ 1, %if.true ]
  %call1 = tail call i32 @f1(i32 %res)
  ret i32 %call1
}

define internal i32 @f2(i32 %x, i32 %y, i32 %z, i1 %cmp.1, i1 %cmp.2) {
; CHECK-LABEL: define internal i32 @f2(
; CHECK-LABEL: entry:
; CHECK-NEXT:    br i1 %cmp.1, label %if.true.1, label %end

; CHECK-LABEL: if.true.1:
; CHECK-NEXT:    br i1 %cmp.2, label %if.true.2, label %end

; CHECK-LABEL: if.true.2:
; CHECK-NEXT:    br label %end

; CHECK-LABEL: end:
; CHECK-NEXT:    %p = phi i32 [ %x, %entry ], [ %y, %if.true.1 ], [ %z, %if.true.2 ]
; CHECK-NEXT:    %c.1 = icmp sgt i32 %p, 5
; CHECK-NEXT:    %c.2 = icmp eq i32 %p, 0
; CHECK-NEXT:    %c.3 = icmp slt i32 %p, 0
; CHECK-NEXT:    %v.1 = select i1 %c.1, i32 10, i32 100
; CHECK-NEXT:    %v.2 = select i1 %c.2, i32 20, i32 200
; CHECK-NEXT:    %v.3 = select i1 %c.3, i32 30, i32 300
; CHECK-NEXT:    %r.1 = add i32 %v.1, %v.2
; CHECK-NEXT:    %r.2 = add i32 %r.1, %v.3
; CHECK-NEXT:    %r.3 = add i32 %r.2, 400
; CHECK-NEXT:    %r.4 = add i32 %r.3, 50
; CHECK-NEXT:    %r.5 = add i32 %r.4, 60
; CHECK-NEXT:    %r.6 = add i32 %r.4, 700
; CHECK-NEXT:    ret i32 %r.6
;
entry:
  br i1 %cmp.1, label %if.true.1, label %end

if.true.1:
  br i1 %cmp.2, label %if.true.2, label %end

if.true.2:
  br label %end

end:
  %p = phi i32 [ %x, %entry ], [ %y, %if.true.1 ], [ %z, %if.true.2 ]
  %c.1 = icmp sgt i32 %p, 5
  %c.2 = icmp eq i32 %p, 0
  %c.3 = icmp slt i32 %p, 0
  %c.4 = icmp sgt i32 %p, 10
  %c.5 = icmp sle i32 %p, 10
  %c.6 = icmp sgt i32 %p, -11
  %c.7 = icmp slt i32 %p, -11
  %v.1 = select i1 %c.1, i32 10, i32 100
  %v.2 = select i1 %c.2, i32 20, i32 200
  %v.3 = select i1 %c.3, i32 30, i32 300
  %v.4 = select i1 %c.4, i32 40, i32 400
  %v.5 = select i1 %c.5, i32 50, i32 500
  %v.6 = select i1 %c.6, i32 60, i32 600
  %v.7 = select i1 %c.7, i32 70, i32 700
  %r.1 = add i32 %v.1, %v.2
  %r.2 = add i32 %r.1, %v.3
  %r.3 = add i32 %r.2, %v.4
  %r.4 = add i32 %r.3, %v.5
  %r.5 = add i32 %r.4, %v.6
  %r.6 = add i32 %r.4, %v.7
  ret i32 %r.6
}

define i32 @caller2(i1 %cmp.1, i1 %cmp.2) {
; CHECK-LABEL: define i32 @caller2(i1 %cmp.1, i1 %cmp.2) {
; CHECK-LABEL: entry:
; CHECK-NEXT:    br i1 %cmp.1, label %if.true, label %end

; CHECK-LABEL: if.true:                                          ; preds = %entry
; CHECK-NEXT:    br label %end

; CHECK-LABEL: end:                                              ; preds = %if.true, %entry
; CHECK-NEXT:    %p1 = phi i32 [ 0, %entry ], [ 1, %if.true ]
; CHECK-NEXT:    %p2 = phi i32 [ 1, %entry ], [ -10, %if.true ]
; CHECK-NEXT:    %p3 = phi i32 [ 1, %entry ], [ 10, %if.true ]
; CHECK-NEXT:    %call1 = tail call i32 @f2(i32 %p1, i32 %p2, i32 %p3, i1 %cmp.1, i1 %cmp.2)
; CHECK-NEXT:    ret i32 %call1
;

entry:
  br i1 %cmp.1, label %if.true, label %end

if.true:
  br label %end

end:
  %p1 = phi i32 [ 0, %entry], [ 1, %if.true ]
  %p2 = phi i32 [ 1, %entry], [ -10, %if.true ]
  %p3 = phi i32 [ 1, %entry], [ 10, %if.true ]
  %call1 = tail call i32 @f2(i32 %p1, i32 %p2, i32 %p3, i1 %cmp.1, i1 %cmp.2)
  ret i32 %call1
}

define internal i32 @f3(i32 %x, i32 %y, i1 %cmp.1) {
; CHECK-LABEL: define internal i32 @f3(i32 %x, i32 %y, i1 %cmp.1) {
; CHECK-LABEL: entry:
; CHECK-NEXT:    br i1 %cmp.1, label %if.true.1, label %end

; CHECK-LABEL: if.true.1:                                        ; preds = %entry
; CHECK-NEXT:    br label %end

; CHECK-LABEL: end:                                              ; preds = %if.true.1, %entry
; CHECK-NEXT:    %p = phi i32 [ %x, %entry ], [ %y, %if.true.1 ]
; CHECK-NEXT:    %c.1 = icmp sgt i32 %p, 5
; CHECK-NEXT:    %c.2 = icmp eq i32 %p, 0
; CHECK-NEXT:    %c.3 = icmp slt i32 %p, 0
; CHECK-NEXT:    %c.4 = icmp sgt i32 %p, 10
; CHECK-NEXT:    %c.5 = icmp sle i32 %p, 10
; CHECK-NEXT:    %c.6 = icmp sgt i32 %p, -11
; CHECK-NEXT:    %c.7 = icmp slt i32 %p, -11
; CHECK-NEXT:    %v.1 = select i1 %c.1, i32 10, i32 100
; CHECK-NEXT:    %v.2 = select i1 %c.2, i32 20, i32 200
; CHECK-NEXT:    %v.3 = select i1 %c.3, i32 30, i32 300
; CHECK-NEXT:    %v.4 = select i1 %c.4, i32 40, i32 400
; CHECK-NEXT:    %v.5 = select i1 %c.5, i32 50, i32 500
; CHECK-NEXT:    %v.6 = select i1 %c.6, i32 60, i32 600
; CHECK-NEXT:    %v.7 = select i1 %c.7, i32 70, i32 700
; CHECK-NEXT:    %r.1 = add i32 %v.1, %v.2
; CHECK-NEXT:    %r.2 = add i32 %r.1, %v.3
; CHECK-NEXT:    %r.3 = add i32 %r.2, %v.4
; CHECK-NEXT:    %r.4 = add i32 %r.3, %v.5
; CHECK-NEXT:    %r.5 = add i32 %r.4, %v.6
; CHECK-NEXT:    %r.6 = add i32 %r.4, %v.7
; CHECK-NEXT:    ret i32 %r.6
;
entry:
  br i1 %cmp.1, label %if.true.1, label %end

if.true.1:
  br label %end

end:
  %p = phi i32 [ %x, %entry ], [ %y, %if.true.1 ]
  %c.1 = icmp sgt i32 %p, 5
  %c.2 = icmp eq i32 %p, 0
  %c.3 = icmp slt i32 %p, 0
  %c.4 = icmp sgt i32 %p, 10
  %c.5 = icmp sle i32 %p, 10
  %c.6 = icmp sgt i32 %p, -11
  %c.7 = icmp slt i32 %p, -11
  %v.1 = select i1 %c.1, i32 10, i32 100
  %v.2 = select i1 %c.2, i32 20, i32 200
  %v.3 = select i1 %c.3, i32 30, i32 300
  %v.4 = select i1 %c.4, i32 40, i32 400
  %v.5 = select i1 %c.5, i32 50, i32 500
  %v.6 = select i1 %c.6, i32 60, i32 600
  %v.7 = select i1 %c.7, i32 70, i32 700
  %r.1 = add i32 %v.1, %v.2
  %r.2 = add i32 %r.1, %v.3
  %r.3 = add i32 %r.2, %v.4
  %r.4 = add i32 %r.3, %v.5
  %r.5 = add i32 %r.4, %v.6
  %r.6 = add i32 %r.4, %v.7
  ret i32 %r.6
}

define i32 @caller3(i32 %y, i1 %cmp.1) {
; CHECK-LABEL: define i32 @caller3(i32 %y, i1 %cmp.1) {
; CHECK-LABEL: entry:
; CHECK-NEXT:    br i1 %cmp.1, label %if.true, label %end

; CHECK-LABEL: if.true:
; CHECK-NEXT:    br label %end

; CHECK-LABEL: end:
; CHECK-NEXT:    %p1 = phi i32 [ 0, %entry ], [ 5, %if.true ]
; CHECK-NEXT:    %call1 = tail call i32 @f3(i32 %p1, i32 %y, i1 %cmp.1)
; CHECK-NEXT:    ret i32 %call1
;
entry:
  br i1 %cmp.1, label %if.true, label %end

if.true:
  br label %end

end:
  %p1 = phi i32 [ 0, %entry], [ 5, %if.true ]
  %call1 = tail call i32 @f3(i32 %p1, i32 %y, i1 %cmp.1)
  ret i32 %call1
}
