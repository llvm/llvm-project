; RUN: opt < %s -simplify-libcalls -S | FileCheck %s
; <rdar://problem/9815881>
; On OSX x86-32, fwrite and fputs aren't called fwrite and fputs.
; Make sure we use the correct names.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.7.2"

%struct.__sFILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__sbuf = type { i8*, i32 }
%struct.__sFILEX = type opaque

@.str = private unnamed_addr constant [13 x i8] c"Hello world\0A\00", align 1
@.str2 = private unnamed_addr constant [3 x i8] c"%s\00", align 1

define void @test1(%struct.__sFILE* %stream) nounwind {
; CHECK: define void @test1
; CHECK: call i32 @"fwrite$UNIX2003"
  %call = tail call i32 (%struct.__sFILE*, i8*, ...)* @fprintf(%struct.__sFILE* %stream, i8* getelementptr inbounds ([13 x i8]* @.str, i32 0, i32 0)) nounwind
  ret void
}

define void @test2(%struct.__sFILE* %stream, i8* %str) nounwind ssp {
; CHECK: define void @test2
; CHECK: call i32 @"fputs$UNIX2003"
  %call = tail call i32 (%struct.__sFILE*, i8*, ...)* @fprintf(%struct.__sFILE* %stream, i8* getelementptr inbounds ([3 x i8]* @.str2, i32 0, i32 0), i8* %str) nounwind
  ret void
}

declare i32 @fprintf(%struct.__sFILE*, i8*, ...) nounwind
