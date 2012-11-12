; RUN: llc < %s | grep ax
; PR2024

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define i32 @foo(i32 %A, i32 %B) nounwind  section ".init.text" {
entry:
	tail call i32 @bar( i32 %A, i32 %B ) nounwind 		; <i32>:0 [#uses=1]
	ret i32 %0
}

declare i32 @bar(i32, i32)
