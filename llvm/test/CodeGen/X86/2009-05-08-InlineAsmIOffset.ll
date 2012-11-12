; RUN: llc < %s -relocation-model=static > %t
; RUN: grep "1: ._pv_cpu_ops+8" %t
; RUN: grep "2: ._G" %t
; PR4152

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"
	%struct.pv_cpu_ops = type { i32, [2 x i32] }
@pv_cpu_ops = external global %struct.pv_cpu_ops		; <%struct.pv_cpu_ops*> [#uses=1]
@G = external global i32		; <i32*> [#uses=1]

define void @x() nounwind {
entry:
	tail call void asm sideeffect "1: $0", "i,~{dirflag},~{fpsr},~{flags}"(i32* getelementptr (%struct.pv_cpu_ops* @pv_cpu_ops, i32 0, i32 1, i32 1)) nounwind
	tail call void asm sideeffect "2: $0", "i,~{dirflag},~{fpsr},~{flags}"(i32* @G) nounwind
	ret void
}
