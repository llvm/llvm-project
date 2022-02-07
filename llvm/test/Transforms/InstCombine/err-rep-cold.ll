; Test the static branch probability heuristics for error-reporting functions.
; RUN: opt < %s -passes=instcombine -S | FileCheck -enable-var-scope %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@stdout = external global %struct._IO_FILE*
@stderr = external global %struct._IO_FILE*
@.str = private unnamed_addr constant [13 x i8] c"an error: %d\00", align 1
@.str1 = private unnamed_addr constant [9 x i8] c"an error\00", align 1

define i32 @test1(i32 %a) #0 {
; CHECK-LABEL: @test1
entry:
  %cmp = icmp sgt i32 %a, 8
  br i1 %cmp, label %if.then, label %return

if.then:                                          ; preds = %entry
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %call = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %0, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0), i32 %a) #1
  br label %return

; CHECK: %call = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %0, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0), i32 %a) #[[$AT1:[0-9]+]]

return:                                           ; preds = %entry, %if.then
  %retval.0 = phi i32 [ 1, %if.then ], [ 0, %entry ]
  ret i32 %retval.0
}

declare i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) #1

define i32 @test2(i32 %a) #0 {
; CHECK-LABEL: @test2
entry:
  %cmp = icmp sgt i32 %a, 8
  br i1 %cmp, label %if.then, label %return

if.then:                                          ; preds = %entry
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %1 = tail call i64 @fwrite(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str1, i64 0, i64 0), i64 8, i64 1, %struct._IO_FILE* %0)
  br label %return

; CHECK: tail call i64 @fwrite(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str1, i64 0, i64 0), i64 8, i64 1, %struct._IO_FILE* %0) #[[$AT2:[0-9]+]]

return:                                           ; preds = %entry, %if.then
  %retval.0 = phi i32 [ 1, %if.then ], [ 0, %entry ]
  ret i32 %retval.0
}

declare i64 @fwrite(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) #1

define i32 @test3(i32 %a) #0 {
; CHECK-LABEL: @test3
entry:
  %cmp = icmp sgt i32 %a, 8
  br i1 %cmp, label %if.then, label %return

if.then:                                          ; preds = %entry
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  %1 = tail call i64 @fwrite(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str1, i64 0, i64 0), i64 8, i64 1, %struct._IO_FILE* %0)
  br label %return

; CHECK-NOT: tail call i64 @fwrite(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str1, i64 0, i64 0), i64 8, i64 1, %struct._IO_FILE* %0) #[[$AT2]]

return:                                           ; preds = %entry, %if.then
  %retval.0 = phi i32 [ 1, %if.then ], [ 0, %entry ]
  ret i32 %retval.0
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }

; CHECK: attributes #[[$AT1]] = { cold nounwind }
; CHECK: attributes #[[$AT2]] = { cold }

