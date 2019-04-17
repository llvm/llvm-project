;; The bitcast cannot be eliminated because byval arguments need
;; the correct type, or at least a type of the correct size.
; RUN: opt < %s -instcombine -S | grep bitcast
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
	%struct.NSRect = type { [4 x float] }

define void @foo(i8* %context) nounwind  {
entry:
	%tmp1 = bitcast i8* %context to %struct.NSRect*		; <%struct.NSRect*> [#uses=1]
	call void (i32, ...) @bar( i32 3, %struct.NSRect* byval align 4  %tmp1 ) nounwind 
	ret void
}

declare void @bar(i32, ...)
