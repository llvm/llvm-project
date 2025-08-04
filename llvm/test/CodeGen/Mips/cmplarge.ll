; RUN: llc -mtriple=mipsel -mattr=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=cmp16

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mipsel--linux-gnu"

%struct.StorablePicture = type { i32, i32, i32, i32 }



define void @getSubImagesLuma(ptr nocapture %s) #0 {
entry:
  %size_y = getelementptr inbounds %struct.StorablePicture, ptr %s, i32 0, i32 1
  %0 = load i32, ptr %size_y, align 4
  %sub = add nsw i32 %0, -1
  %add5 = add nsw i32 %0, 20
  %cmp6 = icmp sgt i32 %add5, -20
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %j.07 = phi i32 [ %inc, %for.body ], [ -20, %entry ]
  %call = tail call i32 @iClip3(i32 0, i32 %sub, i32 %j.07) #2
  %inc = add nsw i32 %j.07, 1
  %1 = load i32, ptr %size_y, align 4
  %add = add nsw i32 %1, 20
  %cmp = icmp slt i32 %inc, %add
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; cmp16: 	.ent	getSubImagesLuma
; cmp16:	.end	getSubImagesLuma
declare i32 @iClip3(...) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
