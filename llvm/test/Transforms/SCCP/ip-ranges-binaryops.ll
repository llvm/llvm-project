; RUN: opt < %s -ipsccp -S | FileCheck %s

; x = [10, 21), y = [100, 201)
; x + y = [110, 221)
define internal i1 @f.add(i32 %x, i32 %y) {
; CHECK-LABEL: define internal i1 @f.add(i32 %x, i32 %y) {
; CHECK-NEXT:    %a.1 = add i32 %x, %y
; CHECK-NEXT:    %c.2 = icmp sgt i32 %a.1, 219
; CHECK-NEXT:    %c.4 = icmp slt i32 %a.1, 111
; CHECK-NEXT:    %c.5 = icmp eq i32 %a.1, 150
; CHECK-NEXT:    %c.6 = icmp slt i32 %a.1, 150
; CHECK-NEXT:    %res.1 = add i1 false, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, false
; CHECK-NEXT:    %res.3 = add i1 %res.2, %c.4
; CHECK-NEXT:    %res.4 = add i1 %res.3, %c.5
; CHECK-NEXT:    %res.5 = add i1 %res.4, %c.6
; CHECK-NEXT:    ret i1 %res.5
;
  %a.1 = add i32 %x, %y
  %c.1 = icmp sgt i32 %a.1, 220
  %c.2 = icmp sgt i32 %a.1, 219
  %c.3 = icmp slt i32 %a.1, 110
  %c.4 = icmp slt i32 %a.1, 111
  %c.5 = icmp eq i32 %a.1, 150
  %c.6 = icmp slt i32 %a.1, 150
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  %res.4 = add i1 %res.3, %c.5
  %res.5 = add i1 %res.4, %c.6
  ret i1 %res.5
}

define i1 @caller.add() {
; CHECK-LABEL:  define i1 @caller.add() {
; CHECK-NEXT:    %call.1 = tail call i1 @f.add(i32 10, i32 100)
; CHECK-NEXT:    %call.2 = tail call i1 @f.add(i32 20, i32 200)
; CHECK-NEXT:    %res = and i1 %call.1, %call.2
; CHECK-NEXT:    ret i1 %res
;
  %call.1 = tail call i1 @f.add(i32 10, i32 100)
  %call.2 = tail call i1 @f.add(i32 20, i32 200)
  %res = and i1 %call.1, %call.2
  ret i1 %res
}


; x = [10, 21), y = [100, 201)
; x - y = [-190, -79)
define internal i1 @f.sub(i32 %x, i32 %y) {
; CHECK-LABEL: define internal i1 @f.sub(i32 %x, i32 %y) {
; CHECK-NEXT:    %a.1 = sub i32 %x, %y
; CHECK-NEXT:    %c.2 = icmp sgt i32 %a.1, -81
; CHECK-NEXT:    %c.4 = icmp slt i32 %a.1, -189
; CHECK-NEXT:    %c.5 = icmp eq i32 %a.1, -150
; CHECK-NEXT:    %c.6 = icmp slt i32 %a.1, -150
; CHECK-NEXT:    %res.1 = add i1 false, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, false
; CHECK-NEXT:    %res.3 = add i1 %res.2, %c.4
; CHECK-NEXT:    %res.4 = add i1 %res.3, %c.5
; CHECK-NEXT:    %res.5 = add i1 %res.4, %c.6
; CHECK-NEXT:    ret i1 %res.5
;
  %a.1 = sub i32 %x, %y
  %c.1 = icmp sgt i32 %a.1, -80
  %c.2 = icmp sgt i32 %a.1, -81
  %c.3 = icmp slt i32 %a.1, -190
  %c.4 = icmp slt i32 %a.1, -189
  %c.5 = icmp eq i32 %a.1, -150
  %c.6 = icmp slt i32 %a.1, -150
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  %res.4 = add i1 %res.3, %c.5
  %res.5 = add i1 %res.4, %c.6
  ret i1 %res.5
}

define i1 @caller.sub() {
; CHECK-LABEL:  define i1 @caller.sub() {
; CHECK-NEXT:    %call.1 = tail call i1 @f.sub(i32 10, i32 100)
; CHECK-NEXT:    %call.2 = tail call i1 @f.sub(i32 20, i32 200)
; CHECK-NEXT:    %res = and i1 %call.1, %call.2
; CHECK-NEXT:    ret i1 %res
;
  %call.1 = tail call i1 @f.sub(i32 10, i32 100)
  %call.2 = tail call i1 @f.sub(i32 20, i32 200)
  %res = and i1 %call.1, %call.2
  ret i1 %res
}

; x = [10, 21), y = [100, 201)
; x * y = [1000, 4001)
define internal i1 @f.mul(i32 %x, i32 %y) {
; CHECK-LABEL: define internal i1 @f.mul(i32 %x, i32 %y) {
; CHECK-NEXT:    %a.1 = mul i32 %x, %y
; CHECK-NEXT:    %c.2 = icmp sgt i32 %a.1, 3999
; CHECK-NEXT:    %c.4 = icmp slt i32 %a.1, 1001
; CHECK-NEXT:    %c.5 = icmp eq i32 %a.1, 1500
; CHECK-NEXT:    %c.6 = icmp slt i32 %a.1, 1500
; CHECK-NEXT:    %res.1 = add i1 false, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, false
; CHECK-NEXT:    %res.3 = add i1 %res.2, %c.4
; CHECK-NEXT:    %res.4 = add i1 %res.3, %c.5
; CHECK-NEXT:    %res.5 = add i1 %res.4, %c.6
; CHECK-NEXT:    ret i1 %res.5
;
  %a.1 = mul i32 %x, %y
  %c.1 = icmp sgt i32 %a.1, 4000
  %c.2 = icmp sgt i32 %a.1, 3999
  %c.3 = icmp slt i32 %a.1, 1000
  %c.4 = icmp slt i32 %a.1, 1001
  %c.5 = icmp eq i32 %a.1, 1500
  %c.6 = icmp slt i32 %a.1, 1500
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  %res.4 = add i1 %res.3, %c.5
  %res.5 = add i1 %res.4, %c.6
  ret i1 %res.5
}

define i1 @caller.mul() {
; CHECK-LABEL:  define i1 @caller.mul() {
; CHECK-NEXT:    %call.1 = tail call i1 @f.mul(i32 10, i32 100)
; CHECK-NEXT:    %call.2 = tail call i1 @f.mul(i32 20, i32 200)
; CHECK-NEXT:    %res = and i1 %call.1, %call.2
; CHECK-NEXT:    ret i1 %res
;
  %call.1 = tail call i1 @f.mul(i32 10, i32 100)
  %call.2 = tail call i1 @f.mul(i32 20, i32 200)
  %res = and i1 %call.1, %call.2
  ret i1 %res
}
