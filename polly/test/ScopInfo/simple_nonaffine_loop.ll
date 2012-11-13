; RUN: opt %loadPolly %defaultOpts -polly-scops -polly-allow-nonaffine -analyze < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.2"

@.str = private unnamed_addr constant [17 x i8] c"Random Value: %d\00", align 1

define i32 @main() nounwind uwtable ssp {
entry:
  %A = alloca [1048576 x i32], align 16
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %0 = phi i32 [ 0, %entry.split ], [ %1, %for.body ]
  %mul = mul i32 %0, 2
  %mul1 = mul nsw i32 %0, %0
  %idxprom1 = zext i32 %mul1 to i64
  %arrayidx = getelementptr inbounds [1048576 x i32]* %A, i64 0, i64 %idxprom1
  store i32 %mul, i32* %arrayidx, align 4
  %1 = add nsw i32 %0, 1
  %exitcond = icmp ne i32 %1, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %call = call i32 @rand() nounwind
  %rem = srem i32 %call, 1024
  %mul2 = shl nsw i32 %rem, 10
  %idxprom3 = sext i32 %mul2 to i64
  %arrayidx4 = getelementptr inbounds [1048576 x i32]* %A, i64 0, i64 %idxprom3
  %2 = load i32* %arrayidx4, align 16
  %call5 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([17 x i8]* @.str, i64 0, i64 0), i32 %2) nounwind
  ret i32 0
}

declare i32 @printf(i8*, ...)

declare i32 @rand()
; CHECK:                { Stmt_for_body[i0] -> MemRef_A[o0] };
