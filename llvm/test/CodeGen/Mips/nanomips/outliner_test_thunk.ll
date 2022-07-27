; RUN: llc -mtriple=nanomips -verify-machineinstrs -enable-machine-outliner < %s | FileCheck %s

define dso_local i32 @testing_thunk1(i32 signext %x, i32 signext %y, i32 signext %a) local_unnamed_addr #0 {
entry:
  %cmp98 = icmp slt i32 %x, %y
  br i1 %cmp98, label %if.then, label %if.else

if.then:                                          ; preds = %entry, %if.then
  %a.tr101 = phi i32 [ %add2, %if.then ], [ %a, %entry ]
  %y.tr100 = phi i32 [ %add, %if.then ], [ %y, %entry ]
  %accumulator.tr99 = phi i32 [ %add4, %if.then ], [ 0, %entry ]
  %mul = mul nsw i32 %a.tr101, 10
  %add = add nsw i32 %mul, %y.tr100
  %reass.add96 = shl i32 %a.tr101, 1
  %add2 = add i32 %add, %reass.add96
  %add3 = add i32 %accumulator.tr99, 13
  %add4 = add i32 %add3, %add
  %cmp = icmp slt i32 %a.tr101, %add
  br i1 %cmp, label %if.then, label %if.else

if.else:                                          ; preds = %if.then, %entry
  %accumulator.tr.lcssa = phi i32 [ 0, %entry ], [ %add4, %if.then ]
  %x.tr.lcssa = phi i32 [ %x, %entry ], [ %a.tr101, %if.then ]
  %y.tr.lcssa = phi i32 [ %y, %entry ], [ %add, %if.then ]
  %a.tr.lcssa = phi i32 [ %a, %entry ], [ %add2, %if.then ]
  %cmp5 = icmp sgt i32 %x.tr.lcssa, %a.tr.lcssa
  br i1 %cmp5, label %if.then6, label %if.else15

common.ret:                                       ; preds = %if.else15, %if.then6
  %common.ret.op = phi i32 [ %accumulator.ret.tr, %if.then6 ], [ %accumulator.ret.tr97, %if.else15 ]
  ret i32 %common.ret.op

if.then6:                                         ; preds = %if.else
  %mul7 = mul nsw i32 %a.tr.lcssa, 10
  %add8 = add nsw i32 %mul7, %y.tr.lcssa
  %reass.add95 = shl i32 %a.tr.lcssa, 1
  %add10 = add i32 %add8, %reass.add95
  %call12 = tail call i32 @testing_thunk1(i32 signext %a.tr.lcssa, i32 signext %add8, i32 signext %add10)
  %add13 = add i32 %add8, -8
  %sub = add i32 %add13, %add10
  %mul14 = mul nsw i32 %call12, %sub
  %accumulator.ret.tr = add nsw i32 %mul14, %accumulator.tr.lcssa
  br label %common.ret

if.else15:                                        ; preds = %if.else
  %mul27.pn = mul nsw i32 %a.tr.lcssa, 10
  %add28.sink = add nsw i32 %mul27.pn, %y.tr.lcssa
  %reass.add93 = shl i32 %a.tr.lcssa, 1
  %add30 = add i32 %add28.sink, %reass.add93
  %add42 = add i32 %add30, %accumulator.tr.lcssa
  %accumulator.ret.tr97 = add i32 %add42, %add28.sink
  br label %common.ret
}

; Function Attrs: nofree nosync nounwind readnone
define dso_local i32 @testing_thunk2(i32 signext %x, i32 signext %y, i32 signext %a) local_unnamed_addr #0 {
entry:
  %cmp = icmp slt i32 %x, %y
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
; CHECK: balc OUTLINED_FUNCTION_0
  %mul = mul nsw i32 %a, 10
  %add = add nsw i32 %mul, %y
  %reass.add85 = shl i32 %a, 1
  %add2 = add i32 %add, %reass.add85
  %call = tail call i32 @testing_thunk1(i32 signext %add, i32 signext %add2, i32 signext %a)
  br label %return

if.else:                                          ; preds = %entry
  %cmp11 = icmp sge i32 %y, %x
  %cmp20 = icmp eq i32 %x, %a
  %or.cond87 = select i1 %cmp20, i1 %cmp11, i1 false
  br i1 %or.cond87, label %if.then21, label %if.end35

if.then21:                                        ; preds = %if.else
  %mul22 = mul nsw i32 %x, 10
  %add23 = add nsw i32 %mul22, %y
  %reass.add82 = shl i32 %x, 1
  %add25 = add i32 %add23, %reass.add82
  %call26 = tail call i32 @testing_thunk1(i32 signext %add23, i32 signext %add25, i32 signext %x)
  br label %return

if.end35:                                         ; preds = %if.else
  %mul5.pn = mul nsw i32 %a, 10
  %add6.sink = add nsw i32 %mul5.pn, %y
  %reass.add8486 = add i32 %add6.sink, %a
  %add36 = shl i32 %reass.add8486, 1
  br label %return

return:                                           ; preds = %if.end35, %if.then21, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %add36, %if.end35 ], [ %call26, %if.then21 ]
  ret i32 %retval.0
}

attributes #0 = { nofree nosync nounwind readnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i7200" "target-features"="+i7200,+soft-float,-noabicalls" "use-soft-float"="true" }

