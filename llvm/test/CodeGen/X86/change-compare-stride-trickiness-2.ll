; RUN: llc < %s
; PR4222

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-pc-linux-gnu"
module asm ".ident\09\22$FreeBSD: head/sys/amd64/amd64/minidump_machdep.c 184499 2008-10-31 10:11:35Z kib $\22"
	%struct.dumperinfo = type <{ i32 (i8*, i8*, i64, i64, i64)*, i8*, i32, i32, i64, i64 }>

define void @minidumpsys(%struct.dumperinfo* %di) nounwind {
entry:
	br label %if.end

if.end:		; preds = %if.end52, %entry
	br label %for.cond.i.preheader

for.cond.i.preheader:		; preds = %if.end52, %if.end
	%indvar688 = phi i64 [ 0, %if.end ], [ %indvar.next689, %if.end52 ]		; <i64> [#uses=3]
	%tmp690 = shl i64 %indvar688, 12		; <i64> [#uses=1]
	%pa.0642 = add i64 %tmp690, 0		; <i64> [#uses=1]
	%indvar688703 = trunc i64 %indvar688 to i32		; <i32> [#uses=1]
	%tmp692693 = add i32 %indvar688703, 1		; <i32> [#uses=1]
	%phitmp = sext i32 %tmp692693 to i64		; <i64> [#uses=1]
	br i1 false, label %if.end52, label %land.lhs.true.i

land.lhs.true.i:		; preds = %for.cond.i.preheader
	%shr2.i = lshr i64 %pa.0642, 18		; <i64> [#uses=0]
	unreachable

if.end52:		; preds = %for.cond.i.preheader
	%phitmp654 = icmp ult i64 %phitmp, 512		; <i1> [#uses=1]
	%indvar.next689 = add i64 %indvar688, 1		; <i64> [#uses=1]
	br i1 %phitmp654, label %for.cond.i.preheader, label %if.end
}

define void @promote(%struct.dumperinfo* %di) nounwind {
entry:
	br label %if.end

if.end:		; preds = %if.end52, %entry
	br label %for.cond.i.preheader

for.cond.i.preheader:		; preds = %if.end52, %if.end
	%indvar688 = phi i32 [ 0, %if.end ], [ %indvar.next689, %if.end52 ]		; <i64> [#uses=3]
	%tmp690 = shl i32 %indvar688, 12		; <i64> [#uses=1]
	%pa.0642 = add i32 %tmp690, 0		; <i64> [#uses=1]
	%tmp692693 = add i32 %indvar688, 1		; <i32> [#uses=1]
	%phitmp = sext i32 %tmp692693 to i64		; <i64> [#uses=1]
	br i1 false, label %if.end52, label %land.lhs.true.i

land.lhs.true.i:		; preds = %for.cond.i.preheader
	%shr2.i = lshr i32 %pa.0642, 18		; <i64> [#uses=0]
	unreachable

if.end52:		; preds = %for.cond.i.preheader
	%phitmp654 = icmp ult i64 %phitmp, 512		; <i1> [#uses=1]
	%indvar.next689 = add i32 %indvar688, 1		; <i64> [#uses=1]
	br i1 %phitmp654, label %for.cond.i.preheader, label %if.end
}
