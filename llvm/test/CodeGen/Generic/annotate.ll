; RUN: llc < %s

; PR15253

@.str = private unnamed_addr constant [4 x i8] c"sth\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [4 x i8] c"t.c\00", section "llvm.metadata"


define i32 @foo(i32 %a) {
entry:
  %0 = call i32 @llvm.annotation.i32(i32 %a, i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str1, i32 0, i32 0), i32 2)
  ret i32 %0
}

declare i32 @llvm.annotation.i32(i32, i8*, i8*, i32) #1
