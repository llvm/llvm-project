; RUN: opt < %s -instcombine -S | FileCheck %s
; Converting the 2 shifts to SHL 6 without the AND is wrong.  PR 8547.

@g_2 = global i32 0, align 4
@.str = constant [10 x i8] c"g_2 = %d\0A\00"

declare i32 @printf(i8*, ...)

define i32 @main() nounwind {
codeRepl:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %codeRepl
  %storemerge = phi i32 [ 0, %codeRepl ], [ 5, %for.cond ]
  store i32 %storemerge, i32* @g_2, align 4
  %shl = shl i32 %storemerge, 30
  %conv2 = lshr i32 %shl, 24
; CHECK:  %0 = shl nuw nsw i32 %storemerge, 6
; CHECK:  %conv2 = and i32 %0, 64
  %tobool = icmp eq i32 %conv2, 0
  br i1 %tobool, label %for.cond, label %codeRepl2

codeRepl2:                                        ; preds = %for.cond
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i64 0, i64 0), i32 %conv2) nounwind
  ret i32 0
}