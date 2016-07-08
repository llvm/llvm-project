; When optimising for size, we don't want to rewrite fputs to fwrite
; because it requires more arguments and thus extra MOVs are required.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@.str = private unnamed_addr constant [10 x i8] c"mylog.txt\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str.2 = private unnamed_addr constant [27 x i8] c"Hello world this is a test\00", align 1

define i32 @main() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: @main(
; CHECK-NOT: call i64 @fwrite
; CHECK: call i32 @fputs

  %call = tail call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i32 0, i32 0)) #2
  %call1 = tail call i32 @fputs(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.2, i32 0, i32 0), %struct._IO_FILE* %call) #2
  ret i32 0
}

declare noalias %struct._IO_FILE* @fopen(i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #1
declare i32 @fputs(i8* nocapture readonly, %struct._IO_FILE* nocapture) local_unnamed_addr #1

attributes #0 = { nounwind optsize }
attributes #1 = { nounwind optsize  }
