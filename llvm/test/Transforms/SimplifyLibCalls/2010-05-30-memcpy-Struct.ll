; RUN: opt -simplify-libcalls %s -S -o - | FileCheck %s
; PR7265

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%union.anon = type { i32, [4 x i8] }

@.str = private constant [3 x i8] c"%s\00"        ; <[3 x i8]*> [#uses=2]

define void @CopyEventArg(%union.anon* %ev) nounwind {
entry:
  %call = call i32 (i8*, i8*, ...)* @sprintf(i8* undef, i8* getelementptr inbounds ([3 x i8]* @.str, i64 0, i64 0), %union.anon* %ev) nounwind ; <i32> [#uses=0]
; CHECK: bitcast %union.anon* %ev to i8*
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
  ret void
}

declare i32 @sprintf(i8*, i8*, ...)

