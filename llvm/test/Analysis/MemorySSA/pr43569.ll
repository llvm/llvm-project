; RUN: opt -pgo-kind=pgo-instr-gen-pipeline -passes="default<O3>" -enable-nontrivial-unswitch -S < %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__profn_c = private constant [1 x i8] c"c"
@b = common dso_local global i32 0, align 4
@a = common dso_local global i16 0, align 2

; CHECK-LABEL: @c()
; Function Attrs: nounwind uwtable
define dso_local void @c() #0 {
entry:
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @__profn_c, i32 0, i32 0), i64 68269137, i32 3, i32 0)
  br label %for.cond

for.cond:                                         ; preds = %for.end, %entry
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @__profn_c, i32 0, i32 0), i64 68269137, i32 3, i32 1)
  store i32 0, i32* @b, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.cond
  %0 = load i32, i32* @b, align 4
  %1 = load i16, i16* @a, align 2
  %conv = sext i16 %1 to i32
  %cmp = icmp slt i32 %0, %conv
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond1
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @__profn_c, i32 0, i32 0), i64 68269137, i32 3, i32 2)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %2 = load i32, i32* @b, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, i32* @b, align 4
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.cond
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(i8*, i64, i32, i32) #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

