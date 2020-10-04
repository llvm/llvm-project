; RUN: opt < %s -domtree -break-crit-edges -analyze -domtree -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='require<domtree>,break-crit-edges,print<domtree>' -disable-output 2>&1| FileCheck %s
; PR932

; CHECK: [3] %brtrue {{{[0-9]+}},{{[0-9]+}}}

declare void @use1(i32)

define void @f(i32 %i, i1 %c) {
entry:
	%A = icmp eq i32 %i, 0		; <i1> [#uses=1]
	br i1 %A, label %brtrue, label %brfalse

brtrue:		; preds = %brtrue, %entry
	%B = phi i1 [ true, %brtrue ], [ false, %entry ]		; <i1> [#uses=1]
	call void @use1( i32 %i )
	br i1 %B, label %brtrue, label %brfalse

brfalse:		; preds = %brtrue, %entry
	call void @use1( i32 %i )
	ret void
}
