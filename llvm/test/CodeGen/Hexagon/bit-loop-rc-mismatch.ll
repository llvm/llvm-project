; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define weak_odr hidden i32 @fred(i32* %this, i32* nocapture readonly dereferenceable(4) %__k) #0 align 2 {
entry:
  %call = tail call i64 @danny(i32* %this, i32* nonnull dereferenceable(4) %__k) #2
  %__p.sroa.0.0.extract.trunc = trunc i64 %call to i32
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %__p.sroa.0.018 = phi i32 [ %call8, %for.body ], [ %__p.sroa.0.0.extract.trunc, %entry ]
  %call8 = tail call i32 @sammy(i32* %this, i32 %__p.sroa.0.018) #2
  %0 = inttoptr i32 %call8 to i32*
  %lnot.i = icmp eq i32* %0, undef
  br i1 %lnot.i, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret i32 0
}

declare hidden i64 @danny(i32*, i32* nocapture readonly dereferenceable(4)) #1 align 2
declare hidden i32 @sammy(i32* nocapture, i32) #0 align 2

attributes #0 = { nounwind optsize "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind optsize readonly "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { optsize }

