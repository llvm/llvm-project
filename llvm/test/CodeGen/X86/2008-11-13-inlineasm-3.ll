; RUN:  llc < %s -mtriple=i686-pc-linux-gnu
; PR 1779
; Using 'A' constraint and a tied constraint together used to crash.
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.linux_dirent64 = type { i64, i64, i16, i8, [0 x i8] }

define i32 @sys_getdents64(i32 %fd, %struct.linux_dirent64* %dirent, i32 %count) {
entry:
	br i1 true, label %cond_next29, label %UnifiedReturnBlock

cond_next29:		; preds = %entry
	%tmp83 = call i32 asm sideeffect "1:\09movl %eax,0($2)\0A2:\09movl %edx,4($2)\0A3:\0A.section .fixup,\22ax\22\0A4:\09movl $3,$0\0A\09jmp 3b\0A.previous\0A .section __ex_table,\22a\22\0A .balign 4 \0A .long 1b,4b\0A .previous\0A .section __ex_table,\22a\22\0A .balign 4 \0A .long 2b,4b\0A .previous\0A", "=r,A,r,i,0,~{dirflag},~{fpsr},~{flags}"(i64 0, i64* null, i32 -14, i32 0) nounwind		; <i32> [#uses=0]
        br label %UnifiedReturnBlock

UnifiedReturnBlock:		; preds = %entry
	ret i32 -14
}
