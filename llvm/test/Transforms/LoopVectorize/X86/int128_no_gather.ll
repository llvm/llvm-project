; RUN: opt < %s -loop-vectorize -S | FileCheck %s

; This test checks that gather/scatter not used for i128 data type.
;CHECK-NOT: gather
;CHECK-NOT: scatter

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = common global [151 x i128] zeroinitializer, align 16
@.str = private unnamed_addr constant [46 x i8] c" PASS.....Y3 1/1 (BUBBLE SORT), X(25) = 5085\0A\00", align 1
@.str.1 = private unnamed_addr constant [44 x i8] c" FAIL.....Y3 1/1 (BUBBLE SORT), X(25) = %d\0A\00", align 1
@str = private unnamed_addr constant [45 x i8] c" PASS.....Y3 1/1 (BUBBLE SORT), X(25) = 5085\00"

; Function Attrs: noinline nounwind uwtable
declare i32 @y3inner() #0 

define i32 @main() local_unnamed_addr #0 {
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %j.0 = phi i128 [ 99999, %entry ], [ %add10, %do.body ]
  %i.0 = phi i128 [ 1, %entry ], [ %add11, %do.body ]
  %and = and i128 %j.0, 32767
  %idxprom = trunc i128 %i.0 to i64
  %arrayidx = getelementptr inbounds [151 x i128], [151 x i128]* @x, i64 0, i64 %idxprom
  store i128 %and, i128* %arrayidx, align 16
  %add = add nuw nsw i128 %j.0, 11111
  %and1 = and i128 %add, 32767
  %add2 = add nuw nsw i128 %i.0, 1
  %idxprom3 = trunc i128 %add2 to i64
  %arrayidx4 = getelementptr inbounds [151 x i128], [151 x i128]* @x, i64 0, i64 %idxprom3
  store i128 %and1, i128* %arrayidx4, align 16
  %add5 = add nuw nsw i128 %j.0, 22222
  %and6 = and i128 %add5, 32767
  %add7 = add nuw nsw i128 %i.0, 2
  %idxprom8 = trunc i128 %add7 to i64
  %arrayidx9 = getelementptr inbounds [151 x i128], [151 x i128]* @x, i64 0, i64 %idxprom8
  store i128 %and6, i128* %arrayidx9, align 16
  %add10 = add nuw nsw i128 %j.0, 33333
  %add11 = add nuw nsw i128 %i.0, 3
  %cmp = icmp slt i128 %add11, 149
  br i1 %cmp, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  store i128 1766649, i128* getelementptr inbounds ([151 x i128], [151 x i128]* @x, i64 0, i64 149), align 16
  store i128 1766649, i128* getelementptr inbounds ([151 x i128], [151 x i128]* @x, i64 0, i64 150), align 16
  %call = tail call i32 @y3inner()
  %0 = load i128, i128* getelementptr inbounds ([151 x i128], [151 x i128]* @x, i64 0, i64 25), align 16
  %cmp12 = icmp eq i128 %0, 5085
  br i1 %cmp12, label %if.then, label %if.else

if.then:                                          ; preds = %do.end
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @str, i64 0, i64 0))
  br label %if.end

if.else:                                          ; preds = %do.end
  %coerce.sroa.0.0.extract.trunc = trunc i128 %0 to i64
  %coerce.sroa.2.0.extract.shift = lshr i128 %0, 64
  %coerce.sroa.2.0.extract.trunc = trunc i128 %coerce.sroa.2.0.extract.shift to i64
  %call14 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str.1, i64 0, i64 0), i64 %coerce.sroa.0.0.extract.trunc, i64 %coerce.sroa.2.0.extract.trunc)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @printf(i8*, ...) #1
; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) #2

attributes #0 = { noinline nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="skylake-avx512" "target-features"="+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+clflushopt,+clwb,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+mpx,+pclmul,+pcommit,+pku,+popcnt,+rdrnd,+rdseed,+rtm,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="skylake-avx512" "target-features"="+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+clflushopt,+clwb,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+lzcnt,+mmx,+movbe,+mpx,+pclmul,+pcommit,+pku,+popcnt,+rdrnd,+rdseed,+rtm,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
