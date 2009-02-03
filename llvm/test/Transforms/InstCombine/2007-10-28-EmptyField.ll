; RUN: llvm-as < %s | opt -instcombine -disable-output
; PR1749

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.__large_struct = type { [100 x i64] }
	%struct.compat_siginfo = type { i32, i32, i32, { [29 x i32] } }
	%struct.siginfo_t = type { i32, i32, i32, { { i32, i32, [0 x i8], %struct.sigval_t, i32 }, [88 x i8] } }
	%struct.sigval_t = type { i8* }

define i32 @copy_siginfo_to_user32(%struct.compat_siginfo* %to, %struct.siginfo_t* %from) {
entry:
	%from_addr = alloca %struct.siginfo_t*		; <%struct.siginfo_t**> [#uses=1]
	%tmp344 = load %struct.siginfo_t** %from_addr, align 8		; <%struct.siginfo_t*> [#uses=1]
	%tmp345 = getelementptr %struct.siginfo_t* %tmp344, i32 0, i32 3		; <{ { i32, i32, [0 x i8], %struct.sigval_t, i32 }, [88 x i8] }*> [#uses=1]
	%tmp346 = getelementptr { { i32, i32, [0 x i8], %struct.sigval_t, i32 }, [88 x i8] }* %tmp345, i32 0, i32 0		; <{ i32, i32, [0 x i8], %struct.sigval_t, i32 }*> [#uses=1]
	%tmp346347 = bitcast { i32, i32, [0 x i8], %struct.sigval_t, i32 }* %tmp346 to { i32, i32, %struct.sigval_t }*		; <{ i32, i32, %struct.sigval_t }*> [#uses=1]
	%tmp348 = getelementptr { i32, i32, %struct.sigval_t }* %tmp346347, i32 0, i32 2		; <%struct.sigval_t*> [#uses=1]
	%tmp349 = getelementptr %struct.sigval_t* %tmp348, i32 0, i32 0		; <i8**> [#uses=1]
	%tmp349350 = bitcast i8** %tmp349 to i32*		; <i32*> [#uses=1]
	%tmp351 = load i32* %tmp349350, align 8		; <i32> [#uses=1]
	%tmp360 = call i32 asm sideeffect "1:\09movl ${1:k},$2\0A2:\0A.section .fixup,\22ax\22\0A3:\09mov $3,$0\0A\09jmp 2b\0A.previous\0A.section __ex_table,\22a\22\0A\09.align 8\0A\09.quad 1b,3b\0A.previous", "=r,ir,*m,i,0,~{dirflag},~{fpsr},~{flags}"( i32 %tmp351, %struct.__large_struct* null, i32 -14, i32 0 )		; <i32> [#uses=0]
	unreachable
}
