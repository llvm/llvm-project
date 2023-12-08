; RUN: llc -march=hexagon -mcpu=hexagonv5 -simplifycfg-require-and-preserve-domtree=1 < %s
; REQUIRES: asserts

define void @test(ptr noalias nocapture readonly %src, i32 %srcStride) local_unnamed_addr #0 {
entry:
  %add.ptr = getelementptr inbounds i8, ptr %src, i32 %srcStride
  %add.ptr2 = getelementptr inbounds i8, ptr %add.ptr, i32 %srcStride
  %add.ptr3 = getelementptr inbounds i8, ptr %add.ptr2, i32 %srcStride
  br label %for.body9.epil

for.body9.epil:
  %inc.sink385.epil = phi i32 [ %add17.epil, %for.body9.epil ], [ 2, %entry ]
  %sr.epil = phi i8 [ %0, %for.body9.epil ], [ undef, %entry ]
  %sr431.epil = phi i8 [ %2, %for.body9.epil ], [ 0, %entry ]
  %sr432.epil = phi i8 [ %sr431.epil, %for.body9.epil ], [ 0, %entry ]
  %epil.iter = phi i32 [ %epil.iter.sub, %for.body9.epil ], [ undef, %entry ]
  %sub11.epil = add i32 %inc.sink385.epil, -1
  %add17.epil = add nuw i32 %inc.sink385.epil, 1
  %conv19.epil = zext i8 %sr.epil to i32
  %add21.epil = add i32 %inc.sink385.epil, 2
  %arrayidx22.epil = getelementptr inbounds i8, ptr %src, i32 %add21.epil
  %0 = load i8, ptr %arrayidx22.epil, align 1
  %conv23.epil = zext i8 %0 to i32
  %1 = load i8, ptr undef, align 1
  %conv42.epil = zext i8 %1 to i32
  %conv53.epil = zext i8 %sr432.epil to i32
  %2 = load i8, ptr undef, align 1
  %conv61.epil = zext i8 %2 to i32
  %3 = load i8, ptr undef, align 1
  %conv65.epil = zext i8 %3 to i32
  %4 = load i8, ptr null, align 1
  %conv69.epil = zext i8 %4 to i32
  %5 = load i8, ptr undef, align 1
  %conv72.epil = zext i8 %5 to i32
  %6 = load i8, ptr undef, align 1
  %conv76.epil = zext i8 %6 to i32
  %7 = load i8, ptr undef, align 1
  %conv80.epil = zext i8 %7 to i32
  %8 = load i8, ptr undef, align 1
  %conv84.epil = zext i8 %8 to i32
  %9 = load i8, ptr undef, align 1
  %conv88.epil = zext i8 %9 to i32
  %10 = load i8, ptr undef, align 1
  %conv91.epil = zext i8 %10 to i32
  %11 = load i8, ptr undef, align 1
  %conv95.epil = zext i8 %11 to i32
  %12 = load i8, ptr undef, align 1
  %conv99.epil = zext i8 %12 to i32
  %add.epil = add nuw nsw i32 0, %conv19.epil
  %add16.epil = add nuw nsw i32 %add.epil, 0
  %add20.epil = add nuw nsw i32 %add16.epil, 0
  %add24.epil = add nuw nsw i32 %add20.epil, 0
  %add28.epil = add nuw nsw i32 %add24.epil, 0
  %add32.epil = add nuw nsw i32 %add28.epil, 0
  %add35.epil = add i32 %add32.epil, 0
  %add39.epil = add i32 %add35.epil, 0
  %add43.epil = add i32 %add39.epil, %conv53.epil
  %add47.epil = add i32 %add43.epil, 0
  %add51.epil = add i32 %add47.epil, 0
  %add54.epil = add i32 %add51.epil, %conv23.epil
  %add58.epil = add i32 %add54.epil, %conv42.epil
  %add62.epil = add i32 %add58.epil, %conv61.epil
  %add66.epil = add i32 %add62.epil, %conv65.epil
  %add70.epil = add i32 %add66.epil, %conv69.epil
  %add73.epil = add i32 %add70.epil, %conv72.epil
  %add77.epil = add i32 %add73.epil, %conv76.epil
  %add81.epil = add i32 %add77.epil, %conv80.epil
  %add85.epil = add i32 %add81.epil, %conv84.epil
  %add89.epil = add i32 %add85.epil, %conv88.epil
  %add92.epil = add i32 %add89.epil, %conv91.epil
  %add96.epil = add i32 %add92.epil, %conv95.epil
  %add100.epil = add i32 %add96.epil, %conv99.epil
  %mul.epil = mul nsw i32 %add100.epil, 2621
  %add101.epil = add nsw i32 %mul.epil, 32768
  %shr369.epil = lshr i32 %add101.epil, 16
  %conv102.epil = trunc i32 %shr369.epil to i8
  %arrayidx103.epil = getelementptr inbounds i8, ptr undef, i32 %inc.sink385.epil
  store i8 %conv102.epil, ptr %arrayidx103.epil, align 1
  %epil.iter.sub = add i32 %epil.iter, -1
  %epil.iter.cmp = icmp eq i32 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %for.end, label %for.body9.epil

for.end:
  unreachable
}

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv5" "unsafe-fp-math"="false" "use-soft-float"="false" }

