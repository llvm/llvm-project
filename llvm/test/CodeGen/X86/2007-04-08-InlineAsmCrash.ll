; RUN: llc < %s
; PR1314

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "x86_64-unknown-linux-gnu"
	%struct.CycleCount = type { i64, i64 }
	%struct.bc_struct = type { i32, i32, i32, i32, %struct.bc_struct*, i8*, i8* }
@_programStartTime = external global %struct.CycleCount		; <%struct.CycleCount*> [#uses=1]

define fastcc i32 @bc_divide(%struct.bc_struct* %n1, %struct.bc_struct* %n2, %struct.bc_struct** %quot, i32 %scale) nounwind {
entry:
	%tmp7.i46 = tail call i64 asm sideeffect ".byte 0x0f,0x31", "={dx},=*{ax},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) getelementptr (%struct.CycleCount, %struct.CycleCount* @_programStartTime, i32 0, i32 1) )		; <i64> [#uses=0]
	%tmp221 = sdiv i32 10, 0		; <i32> [#uses=1]
	tail call fastcc void @_one_mult( i8* null, i32 0, i32 %tmp221, i8* null )
	ret i32 0
}

declare fastcc void @_one_mult(i8*, i32, i32, i8*)
