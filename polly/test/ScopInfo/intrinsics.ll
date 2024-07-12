; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-print-instructions -disable-output < %s 2>&1 | FileCheck %s
;
; Verify that we remove the ignored intrinsics from the instruction list.
;
; CHECK:       Instructions {
; CHECK-NEXT:      store i32 %i.0, ptr %arrayidx, align 4
; CHECK-NEXT:    }
;
;    int A[1024];
;    void func() {
;      for (int i = 0; i < 1024; i++)
;          A[i] = i;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @fun() #0 {
entry:
  %A = alloca [1024 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds [1024 x i32], ptr %A, i64 0, i64 %idxprom
  call void @llvm.donothing()
  store i32 %i.0, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind readnone
declare void @llvm.donothing() #1

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
