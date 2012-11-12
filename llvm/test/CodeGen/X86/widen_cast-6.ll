; RUN: llc < %s -march=x86 -mattr=+sse41 | FileCheck %s
; CHECK: movd

; Test bit convert that requires widening in the operand.

define i32 @return_v2hi() nounwind {
entry:
	%retval12 = bitcast <2 x i16> zeroinitializer to i32		; <i32> [#uses=1]
	ret i32 %retval12
}
