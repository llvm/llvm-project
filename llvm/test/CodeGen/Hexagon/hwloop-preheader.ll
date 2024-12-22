; RUN: llc -march=hexagon -mcpu=hexagonv5 -hexagon-hwloop-preheader < %s
; REQUIRES: asserts

; Test that the preheader is added to the parent loop, otherwise
; we generate an invalid hardware loop.

; Function Attrs: nounwind readonly
define void @test(i16 signext %n) #0 {
entry:
  br i1 undef, label %for.cond4.preheader.preheader.split.us, label %for.end22

for.cond4.preheader.preheader.split.us:
  %0 = sext i16 %n to i32
  br label %for.body9.preheader.us

for.body9.us:
  %indvars.iv = phi i32 [ %indvars.iv.next.7, %for.body9.us ], [ 0, %for.body9.preheader.us ]
  %indvars.iv.next.7 = add i32 %indvars.iv, 8
  %lftr.wideiv.7 = trunc i32 %indvars.iv.next.7 to i16
  %exitcond.7 = icmp slt i16 %lftr.wideiv.7, 0
  br i1 %exitcond.7, label %for.body9.us, label %for.body9.us.ur

for.body9.preheader.us:
  %i.030.us.pmt = phi i32 [ %inc21.us.pmt, %for.end.loopexit.us ], [ 0, %for.cond4.preheader.preheader.split.us ]
  br i1 undef, label %for.body9.us, label %for.body9.us.ur

for.body9.us.ur:
  %exitcond.ur.old = icmp eq i16 undef, %n
  br i1 %exitcond.ur.old, label %for.end.loopexit.us, label %for.body9.us.ur

for.end.loopexit.us:
  %inc21.us.pmt = add i32 %i.030.us.pmt, 1
  %exitcond33 = icmp eq i32 %inc21.us.pmt, %0
  br i1 %exitcond33, label %for.end22, label %for.body9.preheader.us

for.end22:
  ret void
}

attributes #0 = { nounwind readonly "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
