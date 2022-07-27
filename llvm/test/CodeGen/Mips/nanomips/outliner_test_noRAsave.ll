; RUN: llc -mtriple=nanomips -verify-machineinstrs -enable-machine-outliner < %s | FileCheck %s

define dso_local i32 @testing_noRAsave1(i32 signext %x, i32 signext %y, i32 signext %a) local_unnamed_addr #0 {
entry:
  %cmp96 = icmp slt i32 %x, %y
  br i1 %cmp96, label %if.then, label %if.else

if.then:                                          ; preds = %entry, %if.then
  %a.tr99 = phi i32 [ %add2, %if.then ], [ %a, %entry ]
  %y.tr98 = phi i32 [ %add, %if.then ], [ %y, %entry ]
  %accumulator.tr97 = phi i32 [ %add4, %if.then ], [ 0, %entry ]
  %mul = mul nsw i32 %a.tr99, 10
  %add = add nsw i32 %mul, %y.tr98
  %reass.add94 = shl i32 %a.tr99, 1
  %add2 = add i32 %add, %reass.add94
  %add3 = add i32 %accumulator.tr97, 13
  %add4 = add i32 %add3, %add
  %cmp = icmp slt i32 %a.tr99, %add
  br i1 %cmp, label %if.then, label %if.else

if.else:                                          ; preds = %if.then, %entry
  %accumulator.tr.lcssa = phi i32 [ 0, %entry ], [ %add4, %if.then ]
  %x.tr.lcssa = phi i32 [ %x, %entry ], [ %a.tr99, %if.then ]
  %y.tr.lcssa = phi i32 [ %y, %entry ], [ %add, %if.then ]
  %a.tr.lcssa = phi i32 [ %a, %entry ], [ %add2, %if.then ]
  %cmp5 = icmp sgt i32 %x.tr.lcssa, %a.tr.lcssa
  br i1 %cmp5, label %if.then6, label %if.else15

common.ret:                                       ; preds = %if.end40, %if.then6
  %common.ret.op = phi i32 [ %accumulator.ret.tr, %if.then6 ], [ %accumulator.ret.tr95, %if.end40 ]
  ret i32 %common.ret.op

if.then6:                                         ; preds = %if.else
  %mul7 = mul nsw i32 %a.tr.lcssa, 10
  %add8 = add nsw i32 %mul7, %y.tr.lcssa
  %reass.add93 = shl i32 %a.tr.lcssa, 1
  %add10 = add i32 %add8, %reass.add93
  %call12 = tail call i32 @testing_noRAsave1(i32 signext %a.tr.lcssa, i32 signext %add8, i32 signext %add10)
  %add13 = add i32 %add8, -8
  %sub = add i32 %add13, %add10
  %mul14 = mul nsw i32 %call12, %sub
  %accumulator.ret.tr = add nsw i32 %mul14, %accumulator.tr.lcssa
  br label %common.ret

if.else15:                                        ; preds = %if.else
  %cmp16 = icmp slt i32 %y.tr.lcssa, %x.tr.lcssa
  br i1 %cmp16, label %if.then17, label %if.else24

if.then17:                                        ; preds = %if.else15
  %mul18 = mul nsw i32 %a.tr.lcssa, 10
  %add19 = add nsw i32 %mul18, %y.tr.lcssa
  %reass.add92 = shl i32 %a.tr.lcssa, 1
  %add21 = add i32 %add19, %reass.add92
  br label %if.end40

if.else24:                                        ; preds = %if.else15
  %cmp25 = icmp eq i32 %x.tr.lcssa, %a.tr.lcssa
  br i1 %cmp25, label %if.then26, label %if.else33

if.then26:                                        ; preds = %if.else24
  %mul27 = mul nsw i32 %x.tr.lcssa, 10
  %add28 = add nsw i32 %mul27, %y.tr.lcssa
  %reass.add = shl i32 %x.tr.lcssa, 1
  %add30 = add i32 %add28, %reass.add
  br label %if.end40

if.else33:                                        ; preds = %if.else24
  %mul34 = mul nsw i32 %a.tr.lcssa, 10
  %add35 = add nsw i32 %mul34, %y.tr.lcssa
  %call36 = tail call i32 @testing_noRAsave2(i32 signext %add35, i32 signext %y.tr.lcssa, i32 signext %a.tr.lcssa)
  br label %if.end40

if.end40:                                         ; preds = %if.then26, %if.else33, %if.then17
  %y.addr.0 = phi i32 [ %add21, %if.then17 ], [ %add30, %if.then26 ], [ %call36, %if.else33 ]
  %x.addr.0 = phi i32 [ %add19, %if.then17 ], [ %add28, %if.then26 ], [ %add35, %if.else33 ]
  %add41 = add i32 %y.addr.0, %accumulator.tr.lcssa
  %accumulator.ret.tr95 = add i32 %add41, %x.addr.0
  br label %common.ret
}

; Function Attrs: nofree nosync nounwind readnone
define dso_local i32 @testing_noRAsave2(i32 signext %x, i32 signext %y, i32 signext %a) local_unnamed_addr #0 {
entry:
  %cmp = icmp slt i32 %x, %y
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
; CHECK: balc OUTLINED_FUNCTION_0
  %mul = mul nsw i32 %a, 10
  %add = add nsw i32 %mul, %y
  %reass.add85 = shl i32 %a, 1
  %add2 = add i32 %add, %reass.add85
  %call = tail call i32 @testing_noRAsave1(i32 signext %add, i32 signext %add2, i32 signext %a)
  br label %return

if.else:                                          ; preds = %entry
  %cmp3 = icmp sgt i32 %x, %a
  br i1 %cmp3, label %if.then4, label %if.else10

if.then4:                                         ; preds = %if.else
  %mul5 = mul nsw i32 %a, 10
  %add6 = add nsw i32 %mul5, %y
  %reass.add84 = shl i32 %a, 1
  %add8 = add i32 %add6, %reass.add84
  br label %if.end35

if.else10:                                        ; preds = %if.else
  %cmp11 = icmp slt i32 %y, %x
  br i1 %cmp11, label %if.then12, label %if.else21

if.then12:                                        ; preds = %if.else10
; CHECK: balc OUTLINED_FUNCTION_1
  %mul13 = mul nsw i32 %a, 10
  %add14 = add nsw i32 %mul13, %y
  %reass.add83 = shl i32 %a, 1
  %add16 = add i32 %add14, %reass.add83
  %call17 = tail call i32 @testing_noRAsave1(i32 signext %a, i32 signext %add14, i32 signext %add16)
  %add18 = add i32 %add14, -8
  %sub19 = add i32 %add18, %add16
  %mul20 = mul nsw i32 %call17, %sub19
  br label %return

if.else21:                                        ; preds = %if.else10
  %cmp22 = icmp eq i32 %x, %a
  br i1 %cmp22, label %if.then23, label %if.else28

if.then23:                                        ; preds = %if.else21
  %mul24 = mul nsw i32 %x, 10
  %add25 = add nsw i32 %mul24, %y
  %reass.add = shl i32 %x, 1
  %add27 = add i32 %add25, %reass.add
  br label %if.end35

if.else28:                                        ; preds = %if.else21
  %mul29 = mul nsw i32 %a, 10
  %add30 = add nsw i32 %mul29, %y
  %call31 = tail call i32 @testing_noRAsave2(i32 signext %add30, i32 signext %y, i32 signext %a)
  br label %if.end35

if.end35:                                         ; preds = %if.then4, %if.then23, %if.else28
  %x.addr.0 = phi i32 [ %add6, %if.then4 ], [ %add25, %if.then23 ], [ %add30, %if.else28 ]
  %y.addr.0 = phi i32 [ %add8, %if.then4 ], [ %add27, %if.then23 ], [ %call31, %if.else28 ]
  %mul36 = mul nsw i32 %y.addr.0, %x.addr.0
  br label %return

return:                                           ; preds = %if.end35, %if.then12, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %mul36, %if.end35 ], [ %mul20, %if.then12 ]
  ret i32 %retval.0
}

attributes #0 = { nofree nosync nounwind readnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i7200" "target-features"="+i7200,+soft-float,-noabicalls" "use-soft-float"="true" }

