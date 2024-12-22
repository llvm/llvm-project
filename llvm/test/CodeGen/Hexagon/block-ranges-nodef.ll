; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

declare void @foo() #0

define hidden fastcc void @fred(i32 %a, i64 %b, i64 %c) unnamed_addr #1 {
entry:
  %cmp17 = icmp ne i64 %c, 0
  %conv19 = zext i1 %cmp17 to i64
  %or = or i64 %conv19, %b
  store i64 %or, ptr undef, align 8
  br i1 undef, label %if.then44, label %if.end96

if.then44:                                        ; preds = %entry
  br i1 undef, label %overflow, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.then44
  br i1 undef, label %overflow, label %if.end52

if.end52:                                         ; preds = %lor.lhs.false
  br i1 undef, label %if.then55, label %if.end96

if.then55:                                        ; preds = %if.end52
  %cmp60 = icmp slt i32 %a, 0
  %or.cond = or i1 %cmp60, false
  %cmp63 = icmp ule i64 %or, undef
  %.cmp63 = or i1 %cmp63, %or.cond
  call void @foo()
  %or.cond299 = and i1 %.cmp63, undef
  br i1 %or.cond299, label %if.then72, label %if.end73

if.then72:                                        ; preds = %if.then55
  unreachable

if.end73:                                         ; preds = %if.then55
  unreachable

if.end96:                                         ; preds = %if.end52, %entry
  br i1 undef, label %if.end102, label %if.then98

if.then98:                                        ; preds = %if.end96
  br label %if.end102

if.end102:                                        ; preds = %if.then98, %if.end96
  unreachable

overflow:                                         ; preds = %lor.lhs.false, %if.then44
  ret void
}

attributes #0 = { noinline norecurse nounwind }
attributes #1 = { noinline nounwind }

