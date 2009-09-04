; RUN: llvm-as < %s | opt -dse | llvm-dis

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

@g80 = external global i8                         ; <i8*> [#uses=3]

declare signext i8 @foo(i8 signext, i8 signext) nounwind readnone ssp

declare i32 @func68(i32) nounwind readonly ssp

; PR4815
define void @test1(i32 %int32p54) noreturn nounwind ssp {
entry:
  br label %bb

bb:                                               ; preds = %bb, %entry
  %storemerge = phi i8 [ %2, %bb ], [ 1, %entry ] ; <i8> [#uses=1]
  store i8 %storemerge, i8* @g80
  %0 = tail call i32 @func68(i32 1) nounwind ssp  ; <i32> [#uses=1]
  %1 = trunc i32 %0 to i8                         ; <i8> [#uses=1]
  store i8 %1, i8* @g80, align 1
  store i8 undef, i8* @g80, align 1
  %2 = tail call signext i8 @foo(i8 signext undef, i8 signext 1) nounwind ; <i8> [#uses=1]
  br label %bb
}
