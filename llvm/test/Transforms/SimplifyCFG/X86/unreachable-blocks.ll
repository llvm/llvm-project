; RUN: opt -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s -disable-output

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

; PR11825
define void @test1() {
entry:
  br label %return

while_block:                                      ; preds = %and_if_cont2, %and_if_cont
  %newlen = sub i32 %newlen, 1
  %newptr = getelementptr i8, i8* %newptr, i64 1
  %test = icmp sgt i32 %newlen, 0
  br i1 %test, label %and_if1, label %and_if_cont2

and_if1:                                          ; preds = %while_block
  %char = load i8, i8* %newptr
  %test2 = icmp ule i8 %char, 32
  br label %and_if_cont2

and_if_cont2:                                     ; preds = %and_if1, %while_block
  %a18 = phi i1 [ %test, %while_block ], [ %test2, %and_if1 ]
  br i1 %a18, label %while_block, label %return

return:                                           ; preds = %and_if_cont2, %and_if_cont
  ret void
}
