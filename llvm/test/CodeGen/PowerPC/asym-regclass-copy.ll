; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; This tests that the GPRC/GPRC_NOR0 intersection subclass relationship with
; GPRC is handled correctly. When it was not, this test would assert.

@gen_random.last = external unnamed_addr global i64, align 8
@.str = external unnamed_addr constant [4 x i8], align 1

declare double @gen_random(double) #0

declare void @benchmark_heapsort(i32 signext, ptr nocapture) #0

define signext i32 @main(i32 signext %argc, ptr nocapture %argv) #0 {
entry:
  br i1 undef, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %entry
  %cond = phi i32 [ 0, %cond.true ], [ 8000000, %entry ]
  %add = add i32 %cond, 1
  %conv = sext i32 %add to i64
  %mul = shl nsw i64 %conv, 3
  %call1 = tail call noalias ptr @malloc(i64 %mul) #1
  br i1 undef, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %cond.end
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 1, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %add
  br i1 %exitcond, label %for.cond.for.end_crit_edge, label %for.body

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %cond.end
  ret i32 0
}

declare noalias ptr @malloc(i64) #0

declare signext i32 @printf(ptr nocapture, ...) #0

declare void @free(ptr nocapture) #0

declare i64 @strtol(ptr, ptr nocapture, i32 signext) #0

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
