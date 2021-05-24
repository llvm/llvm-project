; RUN: llc -march=hexagon -mcpu=hexagonv5 -verify-machineinstrs < %s

; Test that the name rewriter code doesn't chase the Phi operands for
; Phis that do not occur in the loop that is being pipelined.

define void @test(i32 %srcStride) local_unnamed_addr #0 {
entry:
  br label %for.body

for.body:
  %add.ptr3.pn = phi i8* [ undef, %entry ], [ %src4.0394, %for.end ]
  %src2.0390 = phi i8* [ undef, %entry ], [ %add.ptr3.pn, %for.end ]
  %src4.0394 = getelementptr inbounds i8, i8* %add.ptr3.pn, i32 %srcStride
  %sri414 = load i8, i8* undef, align 1
  br i1 undef, label %for.body9.epil, label %for.body9.preheader.new

for.body9.preheader.new:
  br label %for.body9.epil

for.body9.epil:
  %inc.sink385.epil = phi i32 [ %add17.epil, %for.body9.epil ], [ 2, %for.body ], [ undef, %for.body9.preheader.new ]
  %sr420.epil = phi i8 [ undef, %for.body9.epil ], [ %sri414, %for.body ], [ undef, %for.body9.preheader.new ]
  %sr421.epil = phi i8 [ %sr420.epil, %for.body9.epil ], [ undef, %for.body ], [ undef, %for.body9.preheader.new ]
  %sr422.epil = phi i8 [ %sr421.epil, %for.body9.epil ], [ 0, %for.body ], [ undef, %for.body9.preheader.new ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body9.epil ], [ undef, %for.body9.preheader.new ], [ undef, %for.body ]
  %add17.epil = add nuw i32 %inc.sink385.epil, 1
  %add21.epil = add i32 %inc.sink385.epil, 2
  %arrayidx22.epil = getelementptr inbounds i8, i8* undef, i32 %add21.epil
  %conv27.epil = zext i8 %sr422.epil to i32
  %0 = load i8, i8* null, align 1
  %conv61.epil = zext i8 %0 to i32
  %arrayidx94.epil = getelementptr inbounds i8, i8* %src4.0394, i32 %add17.epil
  %1 = load i8, i8* %arrayidx94.epil, align 1
  %add35.epil = add i32 0, %conv27.epil
  %add39.epil = add i32 %add35.epil, 0
  %add43.epil = add i32 %add39.epil, 0
  %add47.epil = add i32 %add43.epil, 0
  %add51.epil = add i32 %add47.epil, 0
  %add54.epil = add i32 %add51.epil, 0
  %add58.epil = add i32 %add54.epil, 0
  %add62.epil = add i32 %add58.epil, %conv61.epil
  %add66.epil = add i32 %add62.epil, 0
  %add70.epil = add i32 %add66.epil, 0
  %add73.epil = add i32 %add70.epil, 0
  %add77.epil = add i32 %add73.epil, 0
  %add81.epil = add i32 %add77.epil, 0
  %add85.epil = add i32 %add81.epil, 0
  %add89.epil = add i32 %add85.epil, 0
  %add92.epil = add i32 %add89.epil, 0
  %add96.epil = add i32 %add92.epil, 0
  %add100.epil = add i32 %add96.epil, 0
  %mul.epil = mul nsw i32 %add100.epil, 2621
  %add101.epil = add nsw i32 %mul.epil, 32768
  %shr369.epil = lshr i32 %add101.epil, 16
  %conv102.epil = trunc i32 %shr369.epil to i8
  store i8 %conv102.epil, i8* undef, align 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.end, label %for.body9.epil

for.end:
  br label %for.body
}

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv5" "unsafe-fp-math"="false" "use-soft-float"="false" }
