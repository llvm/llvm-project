; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s -check-prefix=DELIN

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"

; for (int i = 0; i < 100; ++i) {
;   int t0 = a[i][i];
;   int t1 = t0 + 1;
;   a[i][5] = t1;
; }
; The subscript 5 in a[i][5] is deliberately an i32, mismatching the types of
; other subscript. DependenceAnalysis before the fix crashed due to this
; mismatch.
define void @i32_subscript([100 x [100 x i32]]* %a, i32* %b) {
; DELIN-LABEL: 'Dependence Analysis' for function 'i32_subscript'
entry:
  br label %for.body

for.body:
; DELIN: da analyze - none!
; DELIN: da analyze - anti [=|<]!
; DELIN: da analyze - none!
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.body ]
  %a.addr = getelementptr [100 x [100 x i32]], [100 x [100 x i32]]* %a, i64 0, i64 %i, i64 %i
  %a.addr.2 = getelementptr [100 x [100 x i32]], [100 x [100 x i32]]* %a, i64 0, i64 %i, i32 5
  %0 = load i32, i32* %a.addr, align 4
  %1 = add i32 %0, 1
  store i32 %1, i32* %a.addr.2, align 4
  %i.inc = add nsw i64 %i, 1
  %exitcond = icmp ne i64 %i.inc, 100
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

;  unsigned i, j;
;  for (i = 1; i < SIZE; i++) {
;    for (j = i; j < SIZE; j++) {
;      a[i][j] = a[i+1][j-1] + 2;
;    }
;  }
;  Extends the previous example to coupled MIV subscripts.


@a = global [10004 x [10004 x i32]] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define void @coupled_miv_type_mismatch(i32 %n) #0 {
; DELIN-LABEL: 'Dependence Analysis' for function 'coupled_miv_type_mismatch'
entry:
  br label %for.cond

; DELIN: da analyze - input [* *]!
; DELIN: da analyze - anti [* *|<]!
; DELIN: da analyze - none!
for.cond:                                         ; preds = %for.inc11, %entry
  %indvars.iv11 = phi i64 [ %indvars.iv.next12, %for.inc11 ], [ 1, %entry ]
  %exitcond14 = icmp ne i64 %indvars.iv11, 10000
  br i1 %exitcond14, label %for.cond1.preheader, label %for.end13

for.cond1.preheader:                              ; preds = %for.cond
  %0 = trunc i64 %indvars.iv11 to i32
  br label %for.cond1

for.cond1:                                        ; preds = %for.cond1.preheader, %for.body3
  %indvars.iv8 = phi i64 [ %indvars.iv11, %for.cond1.preheader ], [ %indvars.iv.next9, %for.body3 ]
  %j.0 = phi i32 [ %inc, %for.body3 ], [ %0, %for.cond1.preheader ]
  %lftr.wideiv = trunc i64 %indvars.iv8 to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 10000
  br i1 %exitcond, label %for.body3, label %for.inc11

for.body3:                                        ; preds = %for.cond1
  %sub = add nsw i32 %j.0, -1
  %idxprom = zext i32 %sub to i64
  %1 = add nuw nsw i64 %indvars.iv11, 1
  %arrayidx5 = getelementptr inbounds [10004 x [10004 x i32]], [10004 x [10004 x i32]]* @a, i64 0, i64 %1, i64 %idxprom
  %2 = load i32, i32* %arrayidx5, align 4
  %add6 = add nsw i32 %2, 2
  %arrayidx10 = getelementptr inbounds [10004 x [10004 x i32]], [10004 x [10004 x i32]]* @a, i64 0, i64 %indvars.iv11, i64 %indvars.iv8
  store i32 %add6, i32* %arrayidx10, align 4
  %indvars.iv.next9 = add nuw nsw i64 %indvars.iv8, 1
  %inc = add nuw nsw i32 %j.0, 1
  br label %for.cond1

for.inc11:                                        ; preds = %for.cond1
  %indvars.iv.next12 = add nuw nsw i64 %indvars.iv11, 1
  br label %for.cond

for.end13:                                        ; preds = %for.cond
  ret void
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0"}
