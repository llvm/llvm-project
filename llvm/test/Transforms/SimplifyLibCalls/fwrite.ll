; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

%FILE = type { i32 }

@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

define i64 @foo(%FILE* %f) {
; CHECK: %retval = call i64 @fwrite
  %retval = call i64 @fwrite(i8* getelementptr inbounds ([1 x i8]* @.str, i64 0, i64 0), i64 1, i64 1, %FILE* %f)
  ret i64 %retval
}

declare i64 @fwrite(i8*, i64, i64, %FILE *)
