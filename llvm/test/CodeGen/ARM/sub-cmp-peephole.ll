; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s

define i32 @f(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: f:
; CHECK: subs
; CHECK-NOT: cmp
  %cmp = icmp sgt i32 %a, %b
  %sub = sub nsw i32 %a, %b
  %sub. = select i1 %cmp, i32 %sub, i32 0
  ret i32 %sub.
}

define i32 @g(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: g:
; CHECK: subs
; CHECK-NOT: cmp
  %cmp = icmp slt i32 %a, %b
  %sub = sub nsw i32 %b, %a
  %sub. = select i1 %cmp, i32 %sub, i32 0
  ret i32 %sub.
}

define i32 @h(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: h:
; CHECK: subs
; CHECK-NOT: cmp
  %cmp = icmp sgt i32 %a, 3
  %sub = sub nsw i32 %a, 3
  %sub. = select i1 %cmp, i32 %sub, i32 %b
  ret i32 %sub.
}

; rdar://11725965
define i32 @i(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK: i:
; CHECK: subs
; CHECK-NOT: cmp
  %cmp = icmp ult i32 %a, %b
  %sub = sub i32 %b, %a
  %sub. = select i1 %cmp, i32 %sub, i32 0
  ret i32 %sub.
}
; If CPSR is live-out, we can't remove cmp if there exists
; a swapped sub.
define i32 @j(i32 %a, i32 %b) nounwind {
entry:
; CHECK: j:
; CHECK: sub
; CHECK: cmp
  %cmp = icmp eq i32 %b, %a
  %sub = sub nsw i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %cmp2 = icmp sgt i32 %b, %a
  %sel = select i1 %cmp2, i32 %sub, i32 %a
  ret i32 %sel

if.else:
  ret i32 %sub
}

; If the sub/rsb instruction is predicated, we can't use the flags.
; <rdar://problem/12263428>
; Test case from MultiSource/Benchmarks/Ptrdist/bc/number.s
; CHECK: bc_raise
; CHECK: rsbeq
; CHECK: cmp
define i32 @bc_raise() nounwind ssp {
entry:
  %val.2.i = select i1 undef, i32 0, i32 undef
  %sub.i = sub nsw i32 0, %val.2.i
  %retval.0.i = select i1 undef, i32 %val.2.i, i32 %sub.i
  %cmp1 = icmp eq i32 %retval.0.i, 0
  br i1 %cmp1, label %land.lhs.true, label %if.end11

land.lhs.true:                                    ; preds = %num2long.exit
  ret i32 17

if.end11:                                         ; preds = %num2long.exit
  ret i32 23
}
