; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Test that the accessSize is set on a post-increment store. If not, an assert
; is triggered in getBaseAndOffset()

%struct.A = type { i8, i32, i32, i32, [10 x i32], [10 x i32], [80 x i32], [80 x i32], [8 x i32], i32, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16 }

; Function Attrs: nounwind
define fastcc void @Decoder_amr(i8 zeroext %mode) #0 {
entry:
  br label %for.cond64.preheader.i

for.cond64.preheader.i:
  %i.1984.i = phi i32 [ 0, %entry ], [ %inc166.i.1, %for.cond64.preheader.i ]
  %inc166.i = add nsw i32 %i.1984.i, 1
  %arrayidx71.i1422.1 = getelementptr inbounds %struct.A, ptr undef, i32 0, i32 7, i32 %inc166.i
  %storemerge800.i.1 = select i1 undef, i32 1310, i32 undef
  %sub156.i.1 = sub nsw i32 0, %storemerge800.i.1
  %sub156.storemerge800.i.1 = select i1 undef, i32 %storemerge800.i.1, i32 %sub156.i.1
  store i32 %sub156.storemerge800.i.1, ptr %arrayidx71.i1422.1, align 4
  store i32 0, ptr undef, align 4
  %inc166.i.1 = add nsw i32 %i.1984.i, 2
  br label %for.cond64.preheader.i

if.end:
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
