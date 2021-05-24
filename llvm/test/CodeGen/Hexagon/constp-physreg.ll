; RUN: llc -O2 -march hexagon < %s
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define signext i16 @foo(i16 signext %var1, i16 signext %var2) #0 {
entry:
  %0 = or i16 %var2, %var1
  %1 = icmp slt i16 %0, 0
  %cmp8 = icmp sgt i16 %var1, %var2
  %or.cond19 = or i1 %1, %cmp8
  br i1 %or.cond19, label %return, label %if.end

if.end:                                           ; preds = %entry
  br label %return

return:                                           ; preds = %if.end, %if.end15, %entry
  %retval.0.reg2mem.0 = phi i16 [ 0, %entry ], [ 32767, %if.end ]
  ret i16 %retval.0.reg2mem.0
}

attributes #0 = { nounwind readnone "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
