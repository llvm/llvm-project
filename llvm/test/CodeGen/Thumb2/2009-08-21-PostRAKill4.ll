; RUN: llc < %s -asm-verbose=false -O3 -relocation-model=pic -frame-pointer=all -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -post-RA-scheduler

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-apple-darwin9"

@.str = external constant [36 x i8], align 1      ; <[36 x i8]*> [#uses=0]
@.str1 = external constant [31 x i8], align 1     ; <[31 x i8]*> [#uses=1]
@.str2 = external constant [4 x i8], align 1      ; <[4 x i8]*> [#uses=1]

declare i32 @getUnknown(i32, ...) nounwind

declare void @llvm.va_start(i8*) nounwind

declare void @llvm.va_end(i8*) nounwind

declare i32 @printf(i8* nocapture, ...) nounwind

define i32 @main() nounwind {
entry:
  %0 = tail call  i32 (i8*, ...) @printf(i8* getelementptr ([31 x i8], [31 x i8]* @.str1, i32 0, i32 0), i32 1, i32 1, i32 1, i32 1, i32 1, i32 1) nounwind ; <i32> [#uses=0]
  %1 = tail call  i32 (i8*, ...) @printf(i8* getelementptr ([31 x i8], [31 x i8]* @.str1, i32 0, i32 0), i32 -128, i32 116, i32 116, i32 -3852, i32 -31232, i32 -1708916736) nounwind ; <i32> [#uses=0]
  %2 = tail call  i32 (i32, ...) @getUnknown(i32 undef, i32 116, i32 116, i32 -3852, i32 -31232, i32 30556, i32 -1708916736) nounwind ; <i32> [#uses=1]
  %3 = tail call  i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @.str2, i32 0, i32 0), i32 %2) nounwind ; <i32> [#uses=0]
  ret i32 0
}
