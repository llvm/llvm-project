; RUN: llc -mtriple=nanomips -verify-machineinstrs -enable-machine-outliner < %s | FileCheck %s

define i32 @testing_outline1(i32 %x, i32 %y, i32 %a) #0 {
entry:
; CHECK: save 16, $fp, $ra
  %cmp1 = icmp sgt i32 %y, 18
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %mul = mul nsw i32 %a, 10
  %add2 = add nsw i32 %mul, %y
  %add5 = add nsw i32 %add2, 11
  br label %if.end39

if.else:                                          ; preds = %entry
  %cmp6 = icmp slt i32 %a, 18
  br i1 %cmp6, label %if.then7, label %if.else13

if.then7:                                         ; preds = %if.else
  %mul8 = mul nsw i32 %a, 10
  %add9 = add nsw i32 %mul8, %y
  %reass.add9597 = add i32 %add9, %a
  %add12 = shl i32 %reass.add9597, 1
  %sub = add nsw i32 %add12, -10
  br label %if.end39

if.else13:                                        ; preds = %if.else
  %cmp14.not = icmp eq i32 %a, 18
  br i1 %cmp14.not, label %if.then24, label %if.then15

if.then15:                                        ; preds = %if.else13
; CHECK: balc OUTLINED_FUNCTION_0
  %mul16 = mul nsw i32 %a, 10
  %add17 = add nsw i32 %mul16, %y
  %reass.add9294 = add i32 %add17, %a
  %add20 = shl i32 %reass.add9294, 1
  %sub21 = add nsw i32 %add20, -22
  br label %if.end39

if.then24:                                        ; preds = %if.else13
  %add26 = add nsw i32 %y, 180
  %reass.add8991 = shl i32 %y, 1
  %sub30 = add i32 %reass.add8991, 376
  br label %if.end39

if.end39:                                         ; preds = %if.then7, %if.then24, %if.then15, %if.then
  %a.addr.0 = phi i32 [ %add5, %if.then ], [ %sub, %if.then7 ], [ %sub21, %if.then15 ], [ %sub30, %if.then24 ]
  %x.addr.1 = phi i32 [ %add2, %if.then ], [ %add9, %if.then7 ], [ %add17, %if.then15 ], [ %add26, %if.then24 ]
  %add40 = add nsw i32 %x.addr.1, %a.addr.0
  ret i32 %add40
}

attributes #0 = { norecurse nounwind optsize readnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i7200" "target-features"="+i7200,+soft-float,-noabicalls" "use-soft-float"="true" }

