; RUN: opt < %s -passes=instcombine -disable-output

; SimplifyDemandedBits should cope with pointer types.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.VEC_rtx_base = type { i32, i32, [1 x %struct.rtx_def*] }
	%struct.VEC_rtx_gc = type { %struct.VEC_rtx_base }
	%struct.block_symbol = type { [3 x %struct.rtunion], %struct.object_block*, i64 }
	%struct.object_block = type { %struct.section*, i32, i64, %struct.VEC_rtx_gc*, %struct.VEC_rtx_gc* }
	%struct.omp_clause_subcode = type { i32 }
	%struct.rtunion = type { i8* }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.section = type { %struct.unnamed_section }
	%struct.u = type { %struct.block_symbol }
	%struct.unnamed_section = type { %struct.omp_clause_subcode, void (i8*)*, i8*, %struct.section* }

define fastcc void @cse_insn(%struct.rtx_def* %insn, %struct.rtx_def* %libcall_insn) nounwind {
entry:
	br i1 undef, label %bb43, label %bb88

bb43:		; preds = %entry
	br label %bb88

bb88:		; preds = %bb43, %entry
	br i1 undef, label %bb95, label %bb107

bb95:		; preds = %bb88
	unreachable

bb107:		; preds = %bb88
	%0 = load i16, i16* undef, align 8		; <i16> [#uses=1]
	%1 = icmp eq i16 %0, 38		; <i1> [#uses=1]
	%src_eqv_here.0 = select i1 %1, %struct.rtx_def* null, %struct.rtx_def* null		; <%struct.rtx_def*> [#uses=1]
	br i1 undef, label %bb127, label %bb125

bb125:		; preds = %bb107
	br i1 undef, label %bb127, label %bb126

bb126:		; preds = %bb125
	br i1 undef, label %bb129, label %bb133

bb127:		; preds = %bb125, %bb107
	unreachable

bb129:		; preds = %bb126
	br label %bb133

bb133:		; preds = %bb129, %bb126
	br i1 undef, label %bb134, label %bb146

bb134:		; preds = %bb133
	unreachable

bb146:		; preds = %bb133
	br i1 undef, label %bb180, label %bb186

bb180:		; preds = %bb146
	%2 = icmp eq %struct.rtx_def* null, null		; <i1> [#uses=1]
	%3 = zext i1 %2 to i8		; <i8> [#uses=1]
	%4 = icmp ne %struct.rtx_def* %src_eqv_here.0, null		; <i1> [#uses=1]
	%5 = zext i1 %4 to i8		; <i8> [#uses=1]
	%toBool181 = icmp ne i8 %3, 0		; <i1> [#uses=1]
	%toBool182 = icmp ne i8 %5, 0		; <i1> [#uses=1]
	%6 = and i1 %toBool181, %toBool182		; <i1> [#uses=1]
	%7 = zext i1 %6 to i8		; <i8> [#uses=1]
	%toBool183 = icmp ne i8 %7, 0		; <i1> [#uses=1]
	br i1 %toBool183, label %bb184, label %bb186

bb184:		; preds = %bb180
	br i1 undef, label %bb185, label %bb186

bb185:		; preds = %bb184
	br label %bb186

bb186:		; preds = %bb185, %bb184, %bb180, %bb146
	br i1 undef, label %bb190, label %bb195

bb190:		; preds = %bb186
	unreachable

bb195:		; preds = %bb186
	unreachable
}
