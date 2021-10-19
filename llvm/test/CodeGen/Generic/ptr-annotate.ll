; RUN: llc < %s

; PR15253

%struct.mystruct = type { i32 }

@.str = private unnamed_addr constant [4 x i8] c"sth\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [4 x i8] c"t.c\00", section "llvm.metadata"

define void @foo() {
entry:
  %m = alloca i8, align 4
  %0 = call i8* @llvm.ptr.annotation.p0i8(i8* %m, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str1, i32 0, i32 0), i32 2, i8* null)
  store i8 1, i8* %0, align 4
  ret void
}

declare i8* @llvm.ptr.annotation.p0i8(i8*, i8*, i8*, i32, i8*) #1
