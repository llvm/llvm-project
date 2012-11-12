; RUN: llc < %s
; PR1748
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @kernel_init(i8* %unused) {
entry:
	call void asm sideeffect "foo ${0:q}", "=*imr"( i64* null )
	ret i32 0
}

