; RUN: llvm-as < %s | llc -march=x86 -relocation-model=static |\
; RUN:   %prcontext 65534 1 | grep movl | count 1
; PR2536

@g_5 = external global i16		; <i16*> [#uses=2]
@g_107 = external global i16		; <i16*> [#uses=1]
@g_229 = external global i32		; <i32*> [#uses=1]
@g_227 = external global i16		; <i16*> [#uses=1]

define i32 @func_54(i32 %p_55, i16 zeroext  %p_56) nounwind  {
entry:
	load i16* @g_5, align 2		; <i16>:0 [#uses=1]
	zext i16 %0 to i32		; <i32>:1 [#uses=1]
	%.mask = and i32 %1, 65534		; <i32> [#uses=1]
	icmp eq i32 %.mask, 0		; <i1>:2 [#uses=1]
	load i32* @g_229, align 4		; <i32>:3 [#uses=1]
	load i16* @g_227, align 2		; <i16>:4 [#uses=1]
	icmp eq i16 %4, 0		; <i1>:5 [#uses=1]
	load i16* @g_5, align 2		; <i16>:6 [#uses=1]
	br label %bb

bb:		; preds = %bb7.preheader, %entry
	%indvar4 = phi i32 [ 0, %entry ], [ %indvar.next5, %bb7.preheader ]		; <i32> [#uses=1]
	%p_56_addr.1.reg2mem.0 = phi i16 [ %p_56, %entry ], [ %p_56_addr.0, %bb7.preheader ]		; <i16> [#uses=2]
	br i1 %2, label %bb7.preheader, label %bb5

bb5:		; preds = %bb
	store i16 %6, i16* @g_107, align 2
	br label %bb7.preheader

bb7.preheader:		; preds = %bb5, %bb
	icmp eq i16 %p_56_addr.1.reg2mem.0, 0		; <i1>:7 [#uses=1]
	%.0 = select i1 %7, i32 1, i32 0		; <i32> [#uses=1]
	urem i32 1, %.0		; <i32>:8 [#uses=1]
	icmp eq i32 %8, 0		; <i1>:9 [#uses=1]
	%.not = xor i1 %9, true		; <i1> [#uses=1]
	%.not1 = xor i1 %5, true		; <i1> [#uses=1]
	%brmerge = or i1 %.not, %.not1		; <i1> [#uses=1]
	%iftmp.6.0 = select i1 %brmerge, i32 1, i32 0		; <i32> [#uses=1]
	mul i32 %iftmp.6.0, %3		; <i32>:10 [#uses=1]
	icmp eq i32 %10, 0		; <i1>:11 [#uses=1]
	%p_56_addr.0 = select i1 %11, i16 %p_56_addr.1.reg2mem.0, i16 1		; <i16> [#uses=1]
	%indvar.next5 = add i32 %indvar4, 1		; <i32> [#uses=2]
	%exitcond6 = icmp eq i32 %indvar.next5, 17		; <i1> [#uses=1]
	br i1 %exitcond6, label %bb25, label %bb

bb25:		; preds = %bb7.preheader
	ret i32 1
}
