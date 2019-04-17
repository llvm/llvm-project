; RUN: opt < %s -loop-reduce -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; LSR shouldn't normalize IV if it can't be denormalized to the original
; expression.  In this testcase, the normalized expression was denormalized to
; an expression different from the original, and we were losing sign extension.

; CHECK:    [[TMP:%[a-z]+]] = trunc i32 {{.*}} to i8
; CHECK:     {{%[a-z0-9]+}} = sext i8 [[TMP]] to i32

@j = common global i32 0, align 4
@c = common global i32 0, align 4
@g = common global i32 0, align 4
@h = common global i8 0, align 1
@d = common global i32 0, align 4
@i = common global i32 0, align 4
@e = common global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"%x\0A\00", align 1
@a = common global i32 0, align 4
@b = common global i16 0, align 2

; Function Attrs: nounwind optsize ssp uwtable
define i32 @main() #0 {
entry:
  store i8 0, i8* @h, align 1
  %0 = load i32, i32* @j, align 4
  %tobool.i = icmp eq i32 %0, 0
  %1 = load i32, i32* @d, align 4
  %cmp3 = icmp sgt i32 %1, -1
  %.lobit = lshr i32 %1, 31
  %.lobit.not = xor i32 %.lobit, 1
  br label %for.body

for.body:                                         ; preds = %entry, %fn3.exit
  %inc9 = phi i8 [ 0, %entry ], [ %inc, %fn3.exit ]
  %conv = sext i8 %inc9 to i32
  br i1 %tobool.i, label %fn3.exit, label %land.rhs.i

land.rhs.i:                                       ; preds = %for.body
  store i32 0, i32* @c, align 4
  br label %fn3.exit

fn3.exit:                                         ; preds = %for.body, %land.rhs.i
  %inc = add i8 %inc9, 1
  %cmp = icmp sgt i8 %inc, -1
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %fn3.exit
  %.lobit.not. = select i1 %cmp3, i32 %.lobit.not, i32 0
  store i32 %conv, i32* @g, align 4
  store i32 %.lobit.not., i32* @i, align 4
  store i8 %inc, i8* @h, align 1
  %conv7 = sext i8 %inc to i32
  %add = add nsw i32 %conv7, %conv
  store i32 %add, i32* @e, align 4
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), i32 %add) #2
  ret i32 0
}

; Function Attrs: nounwind optsize
declare i32 @printf(i8* nocapture readonly, ...) #1

attributes #0 = { nounwind optsize ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind optsize "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind optsize }
