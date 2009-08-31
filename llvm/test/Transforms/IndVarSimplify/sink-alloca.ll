; RUN: llvm-as < %s | opt -indvars | llvm-dis | FileCheck %s
; PR4775

; Indvars shouldn't sink the alloca out of the entry block, even though
; it's not used until after the loop.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 ()* @main to i8*)],
section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define i32 @main() nounwind {
; CHECK: entry:
; CHECK-NEXT: %result.i = alloca i32, align 4
entry:
  %result.i = alloca i32, align 4                 ; <i32*> [#uses=2]
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %call = call i32 @bar() nounwind                ; <i32> [#uses=1]
  %tobool = icmp eq i32 %call, 0                  ; <i1> [#uses=1]
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  volatile store i32 0, i32* %result.i
  %tmp.i = volatile load i32* %result.i           ; <i32> [#uses=0]
  ret i32 0
}

declare i32 @bar()
